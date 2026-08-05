"""Disk-backed dataset storage for large Smart EDA sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterable

import duckdb
import pandas as pd


STORE_ROOT = Path(tempfile.gettempdir()) / "smart_eda_datasets"
LARGE_DATA_ROW_THRESHOLD = 100_000


def _quote_identifier(value: str) -> str:
    """Quote a DuckDB identifier without accepting executable SQL."""
    return '"' + str(value).replace('"', '""') + '"'


def dataset_fingerprint(file_bytes: bytes, sheet_name: str | None = None) -> str:
    digest = hashlib.sha256()
    digest.update(file_bytes)
    digest.update(b"\0")
    digest.update((sheet_name or "").encode("utf-8", errors="replace"))
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetStore:
    """A Parquet-backed dataset with safe, exact DuckDB queries."""

    dataset_id: str
    parquet_path: Path
    metadata_path: Path
    row_count: int
    column_count: int

    @property
    def is_large(self) -> bool:
        return self.row_count >= LARGE_DATA_ROW_THRESHOLD

    def read_dataframe(self, columns: Iterable[str] | None = None) -> pd.DataFrame:
        """Materialize all rows, optionally restricted to selected columns."""
        selected = list(columns) if columns is not None else None
        if selected is not None and not selected:
            return pd.DataFrame(index=range(self.row_count))

        projection = "*" if selected is None else ", ".join(
            _quote_identifier(column) for column in selected
        )
        with duckdb.connect(database=":memory:") as connection:
            return connection.execute(
                f"SELECT {projection} FROM read_parquet(?)",
                [str(self.parquet_path)],
            ).fetch_df()

    def core_metrics(self, include_duplicates: bool = True) -> dict[str, int | None]:
        """Calculate exact core counts, optionally deferring the expensive DISTINCT."""
        with duckdb.connect(database=":memory:") as connection:
            description = connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(self.parquet_path)]
            ).fetchall()
            columns = [row[0] for row in description]
            missing_terms = [
                f"SUM(CASE WHEN {_quote_identifier(column)} IS NULL THEN 1 ELSE 0 END)"
                for column in columns
            ]
            missing_sql = " + ".join(missing_terms) if missing_terms else "0"
            row_count, missing_cells = connection.execute(
                f"SELECT COUNT(*), {missing_sql} FROM read_parquet(?)",
                [str(self.parquet_path)],
            ).fetchone()
            duplicate_rows = None
            if include_duplicates:
                distinct_count = connection.execute(
                    "SELECT COUNT(*) FROM (SELECT DISTINCT * FROM read_parquet(?))",
                    [str(self.parquet_path)],
                ).fetchone()[0]
                duplicate_rows = int(row_count - distinct_count)

        return {
            "rows": int(row_count),
            "columns": len(columns),
            "missing_cells": int(missing_cells or 0),
            "duplicate_rows": duplicate_rows,
        }


def create_dataset_store(
    dataframe: pd.DataFrame,
    file_bytes: bytes,
    source_name: str,
    sheet_name: str | None = None,
) -> DatasetStore:
    """Persist a normalized table as Parquet and return its reusable store."""
    STORE_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_id = dataset_fingerprint(file_bytes, sheet_name)
    parquet_path = STORE_ROOT / f"{dataset_id}.parquet"
    metadata_path = STORE_ROOT / f"{dataset_id}.json"

    if not parquet_path.exists():
        temporary_path = STORE_ROOT / f"{dataset_id}.tmp.parquet"
        dataframe.to_parquet(
            temporary_path,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        temporary_path.replace(parquet_path)

    metadata = {
        "dataset_id": dataset_id,
        "source_name": Path(source_name).name,
        "sheet_name": sheet_name,
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return DatasetStore(
        dataset_id=dataset_id,
        parquet_path=parquet_path,
        metadata_path=metadata_path,
        row_count=metadata["row_count"],
        column_count=metadata["column_count"],
    )


def purge_stale_datasets(max_age_hours: int = 24) -> int:
    """Remove only Smart EDA temporary datasets older than the retention window."""
    if not STORE_ROOT.exists():
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    removed = 0
    for path in STORE_ROOT.iterdir():
        if path.suffix not in {".parquet", ".json"} and not path.name.endswith(
            ".tmp.parquet"
        ):
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if modified < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    return removed

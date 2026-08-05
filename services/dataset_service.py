"""Build the shared dataset context consumed by dashboard pages."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from modules.data_quality import create_dtype_report


@dataclass(frozen=True)
class DatasetContext:
    numeric_cols: list
    categorical_cols: list
    total_rows: int
    total_columns: int
    total_missing_cells: int
    missing_cells_pct: float
    duplicate_rows: int | None
    dtype_report: pd.DataFrame
    quality_analysis: dict | None
    possible_id_cols: list
    high_cardinality_cols: list
    constant_cols: list
    date_like_cols: list
    outlier_report: pd.DataFrame
    missing_report: pd.DataFrame
    unique_report: pd.DataFrame
    smart_recommendations: pd.DataFrame
    outlier_cols_count: int | None


def build_dataset_context(df, dataset_store, state, cardinality_threshold):
    """Prepare cheap metadata and reuse quality results when settings still match."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    total_rows = int(dataset_store.row_count)
    total_columns = int(dataset_store.column_count)

    if state["dataset_core_metrics"] is None:
        if dataset_store.is_large:
            state["dataset_core_metrics"] = dataset_store.core_metrics(
                include_duplicates=False
            )
        else:
            state["dataset_core_metrics"] = {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "missing_cells": int(df.isnull().sum().sum()),
                "duplicate_rows": None,
            }
    core_metrics = state["dataset_core_metrics"]
    total_missing_cells = int(core_metrics["missing_cells"])
    duplicate_rows = core_metrics["duplicate_rows"]
    total_cells = total_rows * total_columns
    missing_cells_pct = (
        round((total_missing_cells / total_cells) * 100, 2) if total_cells else 0
    )

    dtype_report = create_dtype_report(df)
    quality_analysis = state["quality_analysis"]
    if (
        quality_analysis is not None
        and quality_analysis["cardinality_threshold"] != cardinality_threshold
    ):
        state["quality_analysis"] = None
        state["zscore_analysis"] = None
        quality_analysis = None

    if quality_analysis is None:
        return DatasetContext(
            numeric_cols, categorical_cols, total_rows, total_columns,
            total_missing_cells, missing_cells_pct, duplicate_rows, dtype_report,
            None, [], [], [], [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), None,
        )

    outlier_report = quality_analysis["outlier_report"]
    outlier_cols_count = (
        int((outlier_report["Outliers Count"] > 0).sum())
        if not outlier_report.empty else 0
    )
    return DatasetContext(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        total_rows=total_rows,
        total_columns=total_columns,
        total_missing_cells=total_missing_cells,
        missing_cells_pct=missing_cells_pct,
        duplicate_rows=quality_analysis["duplicate_rows"],
        dtype_report=quality_analysis["dtype_report"],
        quality_analysis=quality_analysis,
        possible_id_cols=quality_analysis["possible_id_cols"],
        high_cardinality_cols=quality_analysis["high_cardinality_cols"],
        constant_cols=quality_analysis["constant_cols"],
        date_like_cols=quality_analysis["date_like_cols"],
        outlier_report=outlier_report,
        missing_report=quality_analysis["missing_report"],
        unique_report=quality_analysis["unique_report"],
        smart_recommendations=quality_analysis["smart_recommendations"],
        outlier_cols_count=outlier_cols_count,
    )

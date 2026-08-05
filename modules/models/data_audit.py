"""Shared row-accounting helpers for every modeling workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class ModelDataAudit:
    source_rows: int
    eligible_rows: int
    excluded_rows: int
    excluded_percentage: float
    required_columns: tuple[str, ...]
    exclusion_rule: str
    sampling_applied: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def build_model_data_audit(
    source_df: pd.DataFrame,
    eligible_df: pd.DataFrame,
    required_columns: Iterable[str],
    exclusion_rule: str,
) -> ModelDataAudit:
    source_rows = int(len(source_df))
    eligible_rows = int(len(eligible_df))
    excluded_rows = max(0, source_rows - eligible_rows)
    excluded_percentage = (
        round(excluded_rows / source_rows * 100, 2) if source_rows else 0.0
    )
    return ModelDataAudit(
        source_rows=source_rows,
        eligible_rows=eligible_rows,
        excluded_rows=excluded_rows,
        excluded_percentage=excluded_percentage,
        required_columns=tuple(dict.fromkeys(required_columns)),
        exclusion_rule=exclusion_rule,
    )


def prepare_complete_cases(
    source_df: pd.DataFrame,
    required_columns: Iterable[str],
    exclusion_rule: str = "Rows missing any required model variable are excluded.",
) -> tuple[pd.DataFrame, ModelDataAudit]:
    columns = list(dict.fromkeys(required_columns))
    eligible_df = source_df.loc[:, columns].dropna().copy()
    audit = build_model_data_audit(
        source_df, eligible_df, columns, exclusion_rule=exclusion_rule
    )
    eligible_df.attrs["model_data_audit"] = audit.to_dict()
    return eligible_df, audit


def render_model_data_audit(
    audit: ModelDataAudit | dict,
    title="Model data audit",
    enforce_user_policy=True,
):
    if isinstance(audit, dict):
        audit = ModelDataAudit(**audit)

    st.markdown(f"#### {title}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Source rows", f"{audit.source_rows:,}")
    c2.metric("Eligible rows used", f"{audit.eligible_rows:,}")
    c3.metric(
        "Rows excluded",
        f"{audit.excluded_rows:,}",
        f"{audit.excluded_percentage:.2f}%",
    )
    st.caption(
        f"Rule: {audit.exclusion_rule} No row sampling was applied. "
        "Train/test splitting or cross-validation may allocate eligible rows to "
        "different evaluation roles, but does not silently discard them."
    )
    if audit.eligible_rows == 0:
        st.error("No eligible rows remain for this model configuration.")
        st.stop()
    if (
        enforce_user_policy
        and audit.excluded_rows > 0
        and st.session_state.get("model_missing_policy", "stop")
        != "allow_temporary_exclusion"
    ):
        st.error(
            "Model stopped: this configuration would temporarily exclude "
            f"{audit.excluded_rows:,} row(s). Clean or impute the required values, "
            "select cleaned data, or explicitly allow temporary exclusion above."
        )
        st.stop()
    elif audit.excluded_percentage > 30:
        st.warning(
            "More than 30% of source rows were excluded. Review missingness and "
            "the selected variables before interpreting the model."
        )


def render_train_test_audit(eligible_rows: int, train_rows: int, test_rows: int):
    st.caption(
        f"Eligible rows: {eligible_rows:,} · Training rows: {train_rows:,} · "
        f"Test rows: {test_rows:,} · Accounted for: {train_rows + test_rows:,}."
    )
    if train_rows + test_rows != eligible_rows:
        st.error("Train/test row accounting does not match the eligible dataset.")

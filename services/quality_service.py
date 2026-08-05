"""On-demand, reusable data-quality analysis."""

import time

from modules.data_quality import (
    calculate_outliers_iqr,
    create_dtype_report,
    create_missing_report,
    create_unique_report,
    detect_constant_columns,
    detect_date_like_columns,
    detect_high_cardinality_columns,
    detect_possible_id_columns,
    generate_smart_recommendations,
)


def run_quality_analysis(df, numeric_cols, categorical_cols, cardinality_threshold):
    started_at = time.perf_counter()
    duplicate_mask = df.duplicated()
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_preview = df.loc[duplicate_mask].head(20)
    possible_id_cols = detect_possible_id_columns(df)
    high_cardinality_cols = detect_high_cardinality_columns(df, cardinality_threshold)
    constant_cols = detect_constant_columns(df)
    date_like_cols = detect_date_like_columns(df)
    outlier_report = calculate_outliers_iqr(df, tuple(numeric_cols))
    missing_report = create_missing_report(df)
    dtype_report = create_dtype_report(df)
    unique_report = create_unique_report(df)
    smart_recommendations = generate_smart_recommendations(
        df=df,
        numeric_cols=tuple(numeric_cols),
        categorical_cols=tuple(categorical_cols),
        missing_report=missing_report,
        duplicate_rows=duplicate_rows,
        possible_id_cols=tuple(possible_id_cols),
        high_cardinality_cols=tuple(high_cardinality_cols),
        constant_cols=tuple(constant_cols),
        date_like_cols=tuple(date_like_cols),
        outlier_report=outlier_report,
    )
    return {
        "rows_analyzed": int(df.shape[0]),
        "elapsed_seconds": round(time.perf_counter() - started_at, 2),
        "duplicate_rows": duplicate_rows,
        "duplicate_preview": duplicate_preview,
        "cardinality_threshold": cardinality_threshold,
        "possible_id_cols": possible_id_cols,
        "high_cardinality_cols": high_cardinality_cols,
        "constant_cols": constant_cols,
        "date_like_cols": date_like_cols,
        "outlier_report": outlier_report,
        "missing_report": missing_report,
        "dtype_report": dtype_report,
        "unique_report": unique_report,
        "smart_recommendations": smart_recommendations,
    }

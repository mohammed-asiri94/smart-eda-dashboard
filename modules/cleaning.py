# ============================================================
# modules/cleaning.py
# Data cleaning, imputation, and outlier handling
# ============================================================

import pandas as pd
import numpy as np
from io import BytesIO
import zipfile

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


# ============================================================
# Outlier handling
# ============================================================

def cap_outliers_iqr(df, numeric_cols):
    df_out = df.copy()
    for col in numeric_cols:
        s = df_out[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        df_out[col] = df_out[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
    return df_out


def remove_outlier_rows_iqr(df, numeric_cols):
    df_out = df.copy()
    for col in numeric_cols:
        s = df_out[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df_out = df_out[df_out[col].isna() | df_out[col].between(lower, upper)]
    return df_out


def cap_outliers_zscore(df, numeric_cols, threshold=3.0):
    df_out = df.copy()
    for col in numeric_cols:
        s = df_out[col].dropna()
        if len(s) == 0:
            continue
        mean, std = s.mean(), s.std()
        if std == 0:
            continue
        df_out[col] = df_out[col].clip(
            lower=mean - threshold * std, upper=mean + threshold * std
        )
    return df_out


def remove_outlier_rows_zscore(df, numeric_cols, threshold=3.0):
    df_out = df.copy()
    for col in numeric_cols:
        s = df_out[col].dropna()
        if len(s) == 0:
            continue
        mean, std = s.mean(), s.std()
        if std == 0:
            continue
        z = (df_out[col] - mean).abs() / std
        df_out = df_out[df_out[col].isna() | (z <= threshold)]
    return df_out


# ============================================================
# Imputation helpers
# ============================================================

def apply_group_numeric_imputation(df, numeric_cols, group_col, method):
    df_out = df.copy()
    if not group_col or group_col == "None":
        return df_out
    for col in numeric_cols:
        if method == "Group mean":
            df_out[col] = df_out[col].fillna(
                df_out.groupby(group_col)[col].transform("mean")
            )
            df_out[col] = df_out[col].fillna(df_out[col].mean())
        elif method == "Group median":
            df_out[col] = df_out[col].fillna(
                df_out.groupby(group_col)[col].transform("median")
            )
            df_out[col] = df_out[col].fillna(df_out[col].median())
    return df_out


def apply_group_categorical_imputation(df, categorical_cols, group_col):
    df_out = df.copy()
    if not group_col or group_col == "None":
        return df_out
    for col in categorical_cols:
        if col == group_col:
            continue

        def _mode_fill(series):
            modes = series.mode(dropna=True)
            return series.fillna(modes.iloc[0]) if len(modes) > 0 else series

        df_out[col] = df_out.groupby(group_col)[col].transform(_mode_fill)
        overall = df_out[col].mode(dropna=True)
        if len(overall) > 0:
            df_out[col] = df_out[col].fillna(overall.iloc[0])
    return df_out


def apply_knn_imputation(df, numeric_cols, n_neighbors=5):
    df_out = df.copy()
    usable = [c for c in numeric_cols if df_out[c].notna().sum() > 0]
    if not usable:
        return df_out
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_out[usable] = imputer.fit_transform(df_out[usable])
    return df_out


def apply_iterative_imputation(df, numeric_cols, max_iter=10, random_state=42):
    df_out = df.copy()
    usable = [c for c in numeric_cols if df_out[c].notna().sum() > 0]
    if not usable:
        return df_out
    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    df_out[usable] = imputer.fit_transform(df_out[usable])
    return df_out


def create_multiple_imputed_zip(df, numeric_cols, n_datasets=5, max_iter=10, random_seed=42):
    usable = [c for c in numeric_cols if df[c].notna().sum() > 0]
    if not usable:
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(n_datasets):
            df_imp = df.copy()
            imputer = IterativeImputer(
                max_iter=max_iter, sample_posterior=True, random_state=random_seed + i
            )
            df_imp[usable] = imputer.fit_transform(df_imp[usable])
            zf.writestr(
                f"imputed_dataset_{i + 1}.csv",
                df_imp.to_csv(index=False).encode("utf-8"),
            )
    buf.seek(0)
    return buf


# ============================================================
# Main clean function
# ============================================================

def clean_dataset(
    df,
    remove_duplicates,
    remove_constant_cols,
    remove_id_cols,
    drop_missing_threshold,
    numeric_imputation_method,
    categorical_imputation_method,
    group_col,
    outlier_method,
    possible_id_cols,
    constant_cols,
    numeric_cols,
    categorical_cols,
    knn_neighbors=5,
    iterative_max_iter=10,
    z_score_threshold=3.0,
):
    df_out = df.copy()

    # ── Basic cleaning ──────────────────────────────────────
    if remove_duplicates:
        df_out = df_out.drop_duplicates()

    if remove_constant_cols and constant_cols:
        df_out = df_out.drop(
            columns=[c for c in constant_cols if c in df_out.columns]
        )

    if remove_id_cols and possible_id_cols:
        df_out = df_out.drop(
            columns=[c for c in possible_id_cols if c in df_out.columns]
        )

    if drop_missing_threshold < 100:
        pct = df_out.isnull().mean() * 100
        to_drop = pct[pct > drop_missing_threshold].index.tolist()
        if to_drop:
            df_out = df_out.drop(columns=to_drop)

    # ── Current column lists after dropping ─────────────────
    num_cols = df_out.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_out.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # ── Numeric imputation ───────────────────────────────────
    if numeric_imputation_method == "Mean":
        for col in num_cols:
            df_out[col] = df_out[col].fillna(df_out[col].mean())

    elif numeric_imputation_method == "Median":
        for col in num_cols:
            df_out[col] = df_out[col].fillna(df_out[col].median())

    elif numeric_imputation_method == "Zero":
        for col in num_cols:
            df_out[col] = df_out[col].fillna(0)

    elif numeric_imputation_method in ("Group mean", "Group median"):
        df_out = apply_group_numeric_imputation(
            df_out, num_cols, group_col, numeric_imputation_method
        )

    elif numeric_imputation_method == "KNN":
        df_out = apply_knn_imputation(df_out, num_cols, n_neighbors=knn_neighbors)

    elif numeric_imputation_method == "Iterative":
        df_out = apply_iterative_imputation(
            df_out, num_cols, max_iter=iterative_max_iter
        )

    # ── Categorical imputation ───────────────────────────────
    if categorical_imputation_method == "Mode":
        for col in cat_cols:
            modes = df_out[col].mode(dropna=True)
            if len(modes) > 0:
                df_out[col] = df_out[col].fillna(modes.iloc[0])

    elif categorical_imputation_method == "Unknown":
        for col in cat_cols:
            df_out[col] = df_out[col].fillna("Unknown")

    elif categorical_imputation_method == "Group mode":
        df_out = apply_group_categorical_imputation(df_out, cat_cols, group_col)

    # ── Outlier handling ─────────────────────────────────────
    num_cols_final = df_out.select_dtypes(include=np.number).columns.tolist()

    if outlier_method == "Cap using IQR":
        df_out = cap_outliers_iqr(df_out, num_cols_final)

    elif outlier_method == "Remove rows using IQR":
        df_out = remove_outlier_rows_iqr(df_out, num_cols_final)

    elif outlier_method == "Cap using Z-Score":
        df_out = cap_outliers_zscore(
            df_out, num_cols_final, threshold=z_score_threshold
        )

    elif outlier_method == "Remove rows using Z-Score":
        df_out = remove_outlier_rows_zscore(
            df_out, num_cols_final, threshold=z_score_threshold
        )

    return df_out

# ============================================================
# modules/data_quality.py
# Data quality detection, reports, and smart recommendations
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st


# ============================================================
# Column detection
# ============================================================

@st.cache_data
def detect_possible_id_columns(df):
    possible_id_cols = []
    total_rows = len(df)
    for col in df.columns:
        unique_count = df[col].nunique(dropna=True)
        unique_ratio = unique_count / total_rows if total_rows > 0 else 0
        col_lower = str(col).lower()
        if (
            unique_ratio > 0.90
            or "id" in col_lower
            or "code" in col_lower
            or "number" in col_lower
        ):
            possible_id_cols.append(col)
    return possible_id_cols


@st.cache_data
def detect_high_cardinality_columns(df, threshold=50):
    high_cardinality_cols = []
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    for col in categorical_cols:
        if df[col].nunique(dropna=True) > threshold:
            high_cardinality_cols.append(col)
    return high_cardinality_cols


@st.cache_data
def detect_constant_columns(df):
    return [col for col in df.columns if df[col].nunique(dropna=True) <= 1]


@st.cache_data
def detect_date_like_columns(df):
    date_like_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str).head(100)
            if len(sample) > 0:
                converted = pd.to_datetime(sample, errors="coerce")
                if converted.notna().mean() > 0.70:
                    date_like_cols.append(col)
    return date_like_cols


# ============================================================
# Outlier detection
# ============================================================

@st.cache_data
def calculate_outliers_iqr(df, numeric_cols):
    results = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        results.append({
            "Column": col,
            "Q1": round(q1, 3),
            "Q3": round(q3, 3),
            "IQR": round(iqr, 3),
            "Lower Bound": round(lower, 3),
            "Upper Bound": round(upper, 3),
            "Outliers Count": len(outliers),
            "Outliers Percentage": round((len(outliers) / len(series)) * 100, 2),
        })
    return pd.DataFrame(results)


@st.cache_data
def calculate_outliers_zscore(df, numeric_cols, threshold=3.0):
    results = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        mean = series.mean()
        std = series.std()
        if std == 0:
            continue
        z_scores = (series - mean).abs() / std
        outliers = series[z_scores > threshold]
        results.append({
            "Column": col,
            "Mean": round(mean, 3),
            "Std Dev": round(std, 3),
            "Z-Score Threshold": threshold,
            "Outliers Count": len(outliers),
            "Outliers Percentage": round((len(outliers) / len(series)) * 100, 2),
        })
    return pd.DataFrame(results)


# ============================================================
# Reports
# ============================================================

@st.cache_data
def create_missing_report(df):
    report = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isnull().sum().values,
        "Missing Percentage": (df.isnull().mean().values * 100).round(2),
    })
    return report.sort_values("Missing Percentage", ascending=False)


@st.cache_data
def create_dtype_report(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Non-null Count": df.notnull().sum().values,
        "Missing Count": df.isnull().sum().values,
        "Missing Percentage": (df.isnull().mean().values * 100).round(2),
        "Unique Values": df.nunique(dropna=True).values,
    })


@st.cache_data
def create_unique_report(df):
    total_rows = len(df)
    report = pd.DataFrame({
        "Column": df.columns,
        "Unique Values": df.nunique(dropna=True).values,
        "Unique Percentage": (
            (df.nunique(dropna=True).values / total_rows) * 100
        ).round(2),
    })
    return report.sort_values("Unique Values", ascending=False)


def get_data_quality_status(missing_percentage):
    if missing_percentage < 5:
        return "✅ Good"
    elif missing_percentage <= 30:
        return "⚠️ Warning"
    else:
        return "🔴 Critical"


# ============================================================
# Smart recommendations
# ============================================================

@st.cache_data
def generate_smart_recommendations(
    df,
    numeric_cols,
    categorical_cols,
    missing_report,
    duplicate_rows,
    possible_id_cols,
    high_cardinality_cols,
    constant_cols,
    date_like_cols,
    outlier_report,
):
    recommendations = []

    # Dataset size
    if df.shape[0] < 100:
        recommendations.append({
            "Area": "Dataset Size",
            "Priority": "Medium",
            "Recommendation": (
                "The dataset has fewer than 100 rows. "
                "Be careful when making strong conclusions or training models."
            ),
        })
    else:
        recommendations.append({
            "Area": "Dataset Size",
            "Priority": "Low",
            "Recommendation": "Dataset size looks reasonable for exploratory analysis.",
        })

    # Missing values
    missing_nonzero = missing_report[missing_report["Missing Count"] > 0]
    if missing_nonzero.empty:
        recommendations.append({
            "Area": "Missing Values",
            "Priority": "Low",
            "Recommendation": "No missing values found. Dataset is complete.",
        })
    else:
        for _, row in missing_nonzero.iterrows():
            col = row["Column"]
            pct = row["Missing Percentage"]
            if pct < 5:
                action = "Missingness is low. Consider removing affected rows or imputing."
                priority = "Low"
            elif pct <= 30:
                action = "Moderate missingness. Consider imputation before modeling."
                priority = "Medium"
            else:
                action = "High missingness. Review whether to keep, impute, or drop this column."
                priority = "High"
            if col in numeric_cols:
                method = "For numeric data, median imputation is safer than mean when outliers exist."
            elif col in categorical_cols:
                method = "For categorical data, consider mode imputation or an 'Unknown' category."
            else:
                method = "Review the column type before choosing an imputation method."
            recommendations.append({
                "Area": "Missing Values",
                "Priority": priority,
                "Recommendation": f"Column '{col}' has {pct}% missing. {action} {method}",
            })

    # Duplicates
    if duplicate_rows > 0:
        recommendations.append({
            "Area": "Duplicates",
            "Priority": "Medium",
            "Recommendation": (
                f"Dataset contains {duplicate_rows} duplicate rows. "
                "Remove duplicates before analysis or modeling."
            ),
        })
    else:
        recommendations.append({
            "Area": "Duplicates",
            "Priority": "Low",
            "Recommendation": "No duplicate rows found.",
        })

    # Constant columns
    if constant_cols:
        recommendations.append({
            "Area": "Constant Columns",
            "Priority": "High",
            "Recommendation": (
                f"Columns with only one unique value (not useful for analysis): "
                f"{', '.join(constant_cols)}."
            ),
        })

    # ID columns
    if possible_id_cols:
        recommendations.append({
            "Area": "ID Columns",
            "Priority": "Medium",
            "Recommendation": (
                f"Possible identifier columns (exclude from predictive models): "
                f"{', '.join(possible_id_cols)}."
            ),
        })

    # High cardinality
    if high_cardinality_cols:
        recommendations.append({
            "Area": "High Cardinality",
            "Priority": "Medium",
            "Recommendation": (
                f"High-cardinality categorical columns: {', '.join(high_cardinality_cols)}. "
                "Consider grouping rare categories or using target encoding."
            ),
        })

    # Date columns
    if date_like_cols:
        recommendations.append({
            "Area": "Date Columns",
            "Priority": "Low",
            "Recommendation": (
                f"Date-like columns detected: {', '.join(date_like_cols)}. "
                "Consider converting to datetime and extracting year/month/day features."
            ),
        })

    # Outliers
    if not outlier_report.empty:
        outlier_nonzero = outlier_report[outlier_report["Outliers Count"] > 0]
        if not outlier_nonzero.empty:
            for _, row in outlier_nonzero.iterrows():
                recommendations.append({
                    "Area": "Outliers",
                    "Priority": "Medium",
                    "Recommendation": (
                        f"Column '{row['Column']}' has {row['Outliers Percentage']}% "
                        "possible outliers (IQR method). "
                        "Verify whether these are true extreme values or data entry errors."
                    ),
                })
        else:
            recommendations.append({
                "Area": "Outliers",
                "Priority": "Low",
                "Recommendation": "No outliers detected using the IQR method.",
            })

    recs_df = pd.DataFrame(recommendations)
    recs_df["_order"] = recs_df["Priority"].map({"High": 1, "Medium": 2, "Low": 3})
    recs_df = recs_df.sort_values("_order").drop(columns=["_order"])
    return recs_df

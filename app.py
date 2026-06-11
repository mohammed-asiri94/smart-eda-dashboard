import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import zipfile

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve
)


# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Smart EDA Dashboard",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# Custom CSS
# ============================================================
st.markdown(
    """
    <style>
    .main {
        background-color: #F8FAFC;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero-container {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #38BDF8 100%);
        padding: 35px;
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0px 8px 25px rgba(15, 23, 42, 0.15);
    }

    .hero-title {
        color: white;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #E0F2FE;
        font-size: 18px;
        margin-bottom: 0px;
    }

    .section-card {
        background-color: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0px 4px 14px rgba(15, 23, 42, 0.05);
        margin-bottom: 20px;
    }

    .small-note {
        color: #64748B;
        font-size: 14px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: 800;
        color: #0F172A;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 15px;
        color: #475569;
    }

    section[data-testid="stSidebar"] {
        background-color: #0F172A;
    }

    section[data-testid="stSidebar"] * {
        color: #F8FAFC;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 12px 20px;
        border: 1px solid #E2E8F0;
        color: #0F172A;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2563EB;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# File reading
# ============================================================
def get_excel_sheet_names(uploaded_file):
    file_name = uploaded_file.name.lower()
    if file_name.endswith((".xlsx", ".xls")):
        return pd.ExcelFile(BytesIO(uploaded_file.getvalue())).sheet_names
    return None


@st.cache_data
def load_dataframe(file_bytes, file_name, sheet_name=None):
    if file_name.endswith(".csv"):
        return pd.read_csv(BytesIO(file_bytes))
    elif file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file type. Please upload CSV, XLSX, or XLS file.")


# ============================================================
# Data quality functions
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

    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    for col in categorical_cols:
        unique_count = df[col].nunique(dropna=True)

        if unique_count > threshold:
            high_cardinality_cols.append(col)

    return high_cardinality_cols


@st.cache_data
def detect_constant_columns(df):
    constant_cols = []

    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)

    return constant_cols


@st.cache_data
def detect_date_like_columns(df):
    date_like_cols = []

    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str).head(100)

            if len(sample) > 0:
                converted = pd.to_datetime(sample, errors="coerce")
                success_rate = converted.notna().mean()

                if success_rate > 0.70:
                    date_like_cols.append(col)

    return date_like_cols


@st.cache_data
def calculate_outliers_iqr(df, numeric_cols):
    outlier_results = []

    for col in numeric_cols:
        series = df[col].dropna()

        if len(series) == 0:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series[(series < lower_bound) | (series > upper_bound)]

        outlier_results.append({
            "Column": col,
            "Q1": round(q1, 3),
            "Q3": round(q3, 3),
            "IQR": round(iqr, 3),
            "Lower Bound": round(lower_bound, 3),
            "Upper Bound": round(upper_bound, 3),
            "Outliers Count": len(outliers),
            "Outliers Percentage": round((len(outliers) / len(series)) * 100, 2)
        })

    return pd.DataFrame(outlier_results)


@st.cache_data
def calculate_outliers_zscore(df, numeric_cols, threshold=3.0):
    outlier_results = []

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

        outlier_results.append({
            "Column": col,
            "Mean": round(mean, 3),
            "Std Dev": round(std, 3),
            "Z-Score Threshold": threshold,
            "Outliers Count": len(outliers),
            "Outliers Percentage": round((len(outliers) / len(series)) * 100, 2)
        })

    return pd.DataFrame(outlier_results)


@st.cache_data
def create_missing_report(df):
    missing_report = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isnull().sum().values,
        "Missing Percentage": (df.isnull().mean().values * 100).round(2)
    })

    missing_report = missing_report.sort_values(
        by="Missing Percentage",
        ascending=False
    )

    return missing_report


@st.cache_data
def create_dtype_report(df):
    dtype_report = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Non-null Count": df.notnull().sum().values,
        "Missing Count": df.isnull().sum().values,
        "Missing Percentage": (df.isnull().mean().values * 100).round(2),
        "Unique Values": df.nunique(dropna=True).values
    })

    return dtype_report


@st.cache_data
def create_unique_report(df):
    total_rows = len(df)

    unique_report = pd.DataFrame({
        "Column": df.columns,
        "Unique Values": df.nunique(dropna=True).values,
        "Unique Percentage": ((df.nunique(dropna=True).values / total_rows) * 100).round(2)
    })

    unique_report = unique_report.sort_values(
        by="Unique Values",
        ascending=False
    )

    return unique_report


def get_data_quality_status(missing_percentage):
    if missing_percentage < 5:
        return "Good"
    elif missing_percentage <= 30:
        return "Warning"
    else:
        return "Critical"


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
    outlier_report
):
    recommendations = []

    if df.shape[0] < 100:
        recommendations.append({
            "Area": "Dataset Size",
            "Priority": "Medium",
            "Recommendation": "The dataset has fewer than 100 rows. Be careful when making strong conclusions or training machine learning models."
        })
    else:
        recommendations.append({
            "Area": "Dataset Size",
            "Priority": "Low",
            "Recommendation": "The dataset size looks reasonable for basic exploratory analysis."
        })

    missing_nonzero = missing_report[missing_report["Missing Count"] > 0]

    if missing_nonzero.empty:
        recommendations.append({
            "Area": "Missing Values",
            "Priority": "Low",
            "Recommendation": "No missing values were found. The dataset is complete from a missing-value perspective."
        })
    else:
        for _, row in missing_nonzero.iterrows():
            col = row["Column"]
            missing_percentage = row["Missing Percentage"]

            if missing_percentage < 5:
                action = "Missingness is low. You may remove affected rows or impute values depending on the analysis."
                priority = "Low"
            elif missing_percentage <= 30:
                action = "Moderate missingness detected. Consider imputation before analysis or modeling."
                priority = "Medium"
            else:
                action = "High missingness detected. Review whether this column should be kept, imputed, or removed."
                priority = "High"

            if col in numeric_cols:
                method = "For numeric data, median imputation is often safer than mean imputation when outliers exist."
            elif col in categorical_cols:
                method = "For categorical data, consider filling missing values with the most frequent category or creating an 'Unknown' category."
            else:
                method = "Review the column type before choosing an imputation method."

            recommendations.append({
                "Area": "Missing Values",
                "Priority": priority,
                "Recommendation": f"Column '{col}' has {missing_percentage}% missing values. {action} {method}"
            })

    if duplicate_rows > 0:
        recommendations.append({
            "Area": "Duplicates",
            "Priority": "Medium",
            "Recommendation": f"The dataset contains {duplicate_rows} duplicate rows. Consider removing duplicates before analysis or modeling."
        })
    else:
        recommendations.append({
            "Area": "Duplicates",
            "Priority": "Low",
            "Recommendation": "No duplicate rows were found."
        })

    if constant_cols:
        recommendations.append({
            "Area": "Constant Columns",
            "Priority": "High",
            "Recommendation": f"The following columns have only one unique value and may not be useful for analysis: {', '.join(constant_cols)}."
        })

    if possible_id_cols:
        recommendations.append({
            "Area": "ID Columns",
            "Priority": "Medium",
            "Recommendation": f"The following columns may be identifiers and should usually not be used as predictive features: {', '.join(possible_id_cols)}."
        })

    if high_cardinality_cols:
        recommendations.append({
            "Area": "High Cardinality",
            "Priority": "Medium",
            "Recommendation": f"The following categorical columns have many unique values: {', '.join(high_cardinality_cols)}. Consider grouping rare categories or using suitable encoding methods."
        })

    if date_like_cols:
        recommendations.append({
            "Area": "Date Columns",
            "Priority": "Low",
            "Recommendation": f"The following columns look like dates: {', '.join(date_like_cols)}. Consider converting them to datetime and extracting features such as year, month, or day."
        })

    if not outlier_report.empty:
        outlier_nonzero = outlier_report[outlier_report["Outliers Count"] > 0]

        if not outlier_nonzero.empty:
            for _, row in outlier_nonzero.iterrows():
                recommendations.append({
                    "Area": "Outliers",
                    "Priority": "Medium",
                    "Recommendation": f"Column '{row['Column']}' has {row['Outliers Percentage']}% possible outliers using the IQR method. Check whether these values are true extreme values or data entry errors."
                })
        else:
            recommendations.append({
                "Area": "Outliers",
                "Priority": "Low",
                "Recommendation": "No outliers were detected using the IQR method."
            })

    recommendations_df = pd.DataFrame(recommendations)

    priority_order = {
        "High": 1,
        "Medium": 2,
        "Low": 3
    }

    recommendations_df["Priority Order"] = recommendations_df["Priority"].map(priority_order)

    recommendations_df = recommendations_df.sort_values(
        by="Priority Order"
    ).drop(columns=["Priority Order"])

    return recommendations_df


# ============================================================
# Cleaning and imputation functions
# ============================================================
def cap_outliers_iqr(df, numeric_cols):
    df_capped = df.copy()

    for col in numeric_cols:
        series = df_capped[col].dropna()

        if len(series) == 0:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        df_capped[col] = df_capped[col].clip(
            lower=lower_bound,
            upper=upper_bound
        )

    return df_capped


def remove_outlier_rows_iqr(df, numeric_cols):
    df_removed = df.copy()

    for col in numeric_cols:
        series = df_removed[col].dropna()

        if len(series) == 0:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        df_removed = df_removed[
            (df_removed[col].isna()) |
            ((df_removed[col] >= lower_bound) & (df_removed[col] <= upper_bound))
        ]

    return df_removed


def cap_outliers_zscore(df, numeric_cols, threshold=3.0):
    df_capped = df.copy()

    for col in numeric_cols:
        series = df_capped[col].dropna()

        if len(series) == 0:
            continue

        mean = series.mean()
        std = series.std()

        if std == 0:
            continue

        df_capped[col] = df_capped[col].clip(
            lower=mean - threshold * std,
            upper=mean + threshold * std
        )

    return df_capped


def remove_outlier_rows_zscore(df, numeric_cols, threshold=3.0):
    df_removed = df.copy()

    for col in numeric_cols:
        series = df_removed[col].dropna()

        if len(series) == 0:
            continue

        mean = series.mean()
        std = series.std()

        if std == 0:
            continue

        z_scores = (df_removed[col] - mean).abs() / std

        df_removed = df_removed[
            (df_removed[col].isna()) | (z_scores <= threshold)
        ]

    return df_removed


def apply_group_numeric_imputation(df, numeric_cols, group_col, method):
    df_imputed = df.copy()

    if group_col is None or group_col == "None":
        return df_imputed

    for col in numeric_cols:
        if method == "Group mean":
            df_imputed[col] = df_imputed[col].fillna(
                df_imputed.groupby(group_col)[col].transform("mean")
            )
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())

        elif method == "Group median":
            df_imputed[col] = df_imputed[col].fillna(
                df_imputed.groupby(group_col)[col].transform("median")
            )
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())

    return df_imputed


def apply_group_categorical_imputation(df, categorical_cols, group_col):
    df_imputed = df.copy()

    if group_col is None or group_col == "None":
        return df_imputed

    for col in categorical_cols:
        if col == group_col:
            continue

        def group_mode_fill(series):
            mode_values = series.mode(dropna=True)
            if len(mode_values) > 0:
                return series.fillna(mode_values.iloc[0])
            return series

        df_imputed[col] = df_imputed.groupby(group_col)[col].transform(
            lambda x: group_mode_fill(x)
        )

        overall_mode = df_imputed[col].mode(dropna=True)
        if len(overall_mode) > 0:
            df_imputed[col] = df_imputed[col].fillna(overall_mode.iloc[0])

    return df_imputed


def apply_knn_imputation(df, numeric_cols, n_neighbors=5):
    df_imputed = df.copy()

    usable_numeric_cols = [
        col for col in numeric_cols
        if df_imputed[col].notna().sum() > 0
    ]

    if len(usable_numeric_cols) == 0:
        return df_imputed

    imputer = KNNImputer(n_neighbors=n_neighbors)

    df_imputed[usable_numeric_cols] = imputer.fit_transform(
        df_imputed[usable_numeric_cols]
    )

    return df_imputed


def apply_iterative_imputation(df, numeric_cols, max_iter=10, random_state=42):
    df_imputed = df.copy()

    usable_numeric_cols = [
        col for col in numeric_cols
        if df_imputed[col].notna().sum() > 0
    ]

    if len(usable_numeric_cols) == 0:
        return df_imputed

    imputer = IterativeImputer(
        max_iter=max_iter,
        random_state=random_state
    )

    df_imputed[usable_numeric_cols] = imputer.fit_transform(
        df_imputed[usable_numeric_cols]
    )

    return df_imputed


def create_multiple_imputed_zip(
    df,
    numeric_cols,
    n_datasets=5,
    max_iter=10,
    random_seed=42
):
    usable_numeric_cols = [
        col for col in numeric_cols
        if df[col].notna().sum() > 0
    ]

    if len(usable_numeric_cols) == 0:
        return None

    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for i in range(n_datasets):
            df_imputed = df.copy()

            imputer = IterativeImputer(
                max_iter=max_iter,
                sample_posterior=True,
                random_state=random_seed + i
            )

            df_imputed[usable_numeric_cols] = imputer.fit_transform(
                df_imputed[usable_numeric_cols]
            )

            csv_bytes = df_imputed.to_csv(index=False).encode("utf-8")

            zip_file.writestr(
                f"multiple_imputed_dataset_{i + 1}.csv",
                csv_bytes
            )

    zip_buffer.seek(0)
    return zip_buffer


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
    knn_neighbors,
    iterative_max_iter,
    z_score_threshold=3.0
):
    df_cleaned = df.copy()

    if remove_duplicates:
        df_cleaned = df_cleaned.drop_duplicates()

    if remove_constant_cols and constant_cols:
        cols_to_drop = [col for col in constant_cols if col in df_cleaned.columns]
        df_cleaned = df_cleaned.drop(columns=cols_to_drop)

    if remove_id_cols and possible_id_cols:
        cols_to_drop = [col for col in possible_id_cols if col in df_cleaned.columns]
        df_cleaned = df_cleaned.drop(columns=cols_to_drop)

    if drop_missing_threshold < 100:
        missing_percentages = df_cleaned.isnull().mean() * 100
        cols_to_drop = missing_percentages[
            missing_percentages > drop_missing_threshold
        ].index.tolist()

        if cols_to_drop:
            df_cleaned = df_cleaned.drop(columns=cols_to_drop)

    current_numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
    current_categorical_cols = df_cleaned.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    if numeric_imputation_method == "Mean":
        for col in current_numeric_cols:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())

    elif numeric_imputation_method == "Median":
        for col in current_numeric_cols:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    elif numeric_imputation_method == "Zero":
        for col in current_numeric_cols:
            df_cleaned[col] = df_cleaned[col].fillna(0)

    elif numeric_imputation_method in ["Group mean", "Group median"]:
        df_cleaned = apply_group_numeric_imputation(
            df_cleaned,
            current_numeric_cols,
            group_col,
            numeric_imputation_method
        )

    elif numeric_imputation_method == "KNN":
        df_cleaned = apply_knn_imputation(
            df_cleaned,
            current_numeric_cols,
            n_neighbors=knn_neighbors
        )

    elif numeric_imputation_method == "Iterative":
        df_cleaned = apply_iterative_imputation(
            df_cleaned,
            current_numeric_cols,
            max_iter=iterative_max_iter
        )

    if categorical_imputation_method == "Mode":
        for col in current_categorical_cols:
            mode_values = df_cleaned[col].mode(dropna=True)
            if len(mode_values) > 0:
                df_cleaned[col] = df_cleaned[col].fillna(mode_values.iloc[0])

    elif categorical_imputation_method == "Unknown":
        for col in current_categorical_cols:
            df_cleaned[col] = df_cleaned[col].fillna("Unknown")

    elif categorical_imputation_method == "Group mode":
        df_cleaned = apply_group_categorical_imputation(
            df_cleaned,
            current_categorical_cols,
            group_col
        )

    current_numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()

    if outlier_method == "Cap using IQR":
        df_cleaned = cap_outliers_iqr(df_cleaned, current_numeric_cols)

    elif outlier_method == "Remove rows using IQR":
        df_cleaned = remove_outlier_rows_iqr(df_cleaned, current_numeric_cols)

    elif outlier_method == "Cap using Z-Score":
        df_cleaned = cap_outliers_zscore(df_cleaned, current_numeric_cols, threshold=z_score_threshold)

    elif outlier_method == "Remove rows using Z-Score":
        df_cleaned = remove_outlier_rows_zscore(df_cleaned, current_numeric_cols, threshold=z_score_threshold)

    return df_cleaned


# ============================================================
# Modeling functions
# ============================================================
def quote_column(col):
    col = str(col).replace('"', '\\"')
    return f'Q("{col}")'


def build_model_formula(target_col, predictor_cols, df):
    target_part = quote_column(target_col)

    formula_parts = []

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    for col in predictor_cols:
        if col in categorical_columns:
            formula_parts.append(f"C({quote_column(col)})")
        else:
            formula_parts.append(quote_column(col))

    formula = target_part + " ~ " + " + ".join(formula_parts)

    return formula


def prepare_model_data(df, target_col, predictor_cols):
    selected_cols = [target_col] + predictor_cols
    model_df = df[selected_cols].copy()
    model_df = model_df.dropna()

    return model_df


def make_coefficients_table(model_result, model_type):
    params = model_result.params
    conf = model_result.conf_int()
    pvalues = model_result.pvalues
    bse = model_result.bse

    coef_table = pd.DataFrame({
        "Term": params.index,
        "Coefficient": params.values,
        "Std Error": bse.values,
        "P-value": pvalues.values,
        "CI Lower": conf[0].values,
        "CI Upper": conf[1].values
    })

    coef_table["Coefficient"] = coef_table["Coefficient"].round(4)
    coef_table["Std Error"] = coef_table["Std Error"].round(4)
    coef_table["P-value"] = coef_table["P-value"].round(5)
    coef_table["CI Lower"] = coef_table["CI Lower"].round(4)
    coef_table["CI Upper"] = coef_table["CI Upper"].round(4)

    if model_type == "Binary Logistic Regression":
        coef_table["Odds Ratio"] = np.exp(params.values).round(4)
        coef_table["OR CI Lower"] = np.exp(conf[0].values).round(4)
        coef_table["OR CI Upper"] = np.exp(conf[1].values).round(4)

    if model_type in ["Poisson Regression", "Negative Binomial Regression"]:
        coef_table["IRR"] = np.exp(params.values).round(4)
        coef_table["IRR CI Lower"] = np.exp(conf[0].values).round(4)
        coef_table["IRR CI Upper"] = np.exp(conf[1].values).round(4)

    return coef_table


def run_statistical_model(model_df, target_col, predictor_cols, model_type):
    formula = build_model_formula(target_col, predictor_cols, model_df)

    if model_type == "Linear Regression":
        model_result = smf.ols(
            formula=formula,
            data=model_df
        ).fit()

    elif model_type == "Binary Logistic Regression":
        target_unique = model_df[target_col].dropna().unique()

        if len(target_unique) != 2:
            raise ValueError(
                "Binary Logistic Regression requires the target variable to have exactly 2 unique values."
            )

        model_df = model_df.copy()

        if not set(model_df[target_col].dropna().unique()).issubset({0, 1}):
            sorted_classes = sorted(model_df[target_col].dropna().unique())
            mapping = {
                sorted_classes[0]: 0,
                sorted_classes[1]: 1
            }
            model_df[target_col] = model_df[target_col].map(mapping)

        formula = build_model_formula(target_col, predictor_cols, model_df)

        model_result = smf.logit(
            formula=formula,
            data=model_df
        ).fit(disp=False)

    elif model_type == "Poisson Regression":
        if not pd.api.types.is_numeric_dtype(model_df[target_col]):
            raise ValueError("Poisson Regression requires a numeric count target variable.")

        if (model_df[target_col] < 0).any():
            raise ValueError("Poisson Regression requires non-negative count values.")

        model_result = smf.glm(
            formula=formula,
            data=model_df,
            family=sm.families.Poisson()
        ).fit()

    elif model_type == "Negative Binomial Regression":
        if not pd.api.types.is_numeric_dtype(model_df[target_col]):
            raise ValueError("Negative Binomial Regression requires a numeric count target variable.")

        if (model_df[target_col] < 0).any():
            raise ValueError("Negative Binomial Regression requires non-negative count values.")

        model_result = smf.glm(
            formula=formula,
            data=model_df,
            family=sm.families.NegativeBinomial()
        ).fit()

    else:
        raise ValueError("Unsupported model type.")

    return model_result, model_df, formula


def create_text_download(text, file_name, label):
    st.download_button(
        label=label,
        data=text,
        file_name=file_name,
        mime="text/plain",
        use_container_width=True
    )


# ============================================================
# HTML report and downloads
# ============================================================
def generate_html_report(
    uploaded_file_name,
    selected_sheet,
    total_rows,
    total_columns,
    total_missing_cells,
    missing_cells_percentage,
    duplicate_rows,
    numeric_cols,
    categorical_cols,
    possible_id_cols,
    high_cardinality_cols,
    constant_cols,
    date_like_cols,
    outlier_columns_count,
    missing_report,
    dtype_report,
    unique_report,
    outlier_report,
    smart_recommendations
):
    outlier_html = (
        outlier_report.to_html(index=False)
        if not outlier_report.empty
        else "<p>No numeric columns available for outlier detection.</p>"
    )

    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Smart EDA Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #F8FAFC;
                color: #0F172A;
            }}

            .header {{
                background: linear-gradient(135deg, #1E3A8A, #2563EB, #38BDF8);
                color: white;
                padding: 30px;
                border-radius: 16px;
                margin-bottom: 30px;
            }}

            .header h1 {{
                margin: 0;
                font-size: 36px;
            }}

            .header p {{
                margin-top: 8px;
                color: #E0F2FE;
            }}

            .card {{
                background-color: white;
                padding: 20px;
                border-radius: 14px;
                margin-bottom: 25px;
                border: 1px solid #E2E8F0;
                box-shadow: 0px 4px 12px rgba(15, 23, 42, 0.05);
            }}

            h2 {{
                color: #1E3A8A;
                border-bottom: 2px solid #E2E8F0;
                padding-bottom: 8px;
            }}

            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 15px;
                font-size: 14px;
            }}

            th, td {{
                border: 1px solid #CBD5E1;
                padding: 8px;
                text-align: left;
            }}

            th {{
                background-color: #E0F2FE;
                color: #0F172A;
            }}

            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin-top: 20px;
            }}

            .metric {{
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 12px;
                padding: 15px;
                text-align: center;
            }}

            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2563EB;
            }}

            .metric-label {{
                font-size: 13px;
                color: #64748B;
                margin-top: 5px;
            }}
        </style>
    </head>

    <body>
        <div class="header">
            <h1>Smart EDA Report</h1>
            <p>Automatic exploratory data analysis and data quality summary</p>
        </div>

        <div class="card">
            <h2>Dataset Summary</h2>
            <p><strong>File name:</strong> {uploaded_file_name}</p>
            <p><strong>Excel sheet:</strong> {selected_sheet if selected_sheet is not None else "Not applicable"}</p>

            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value">{total_rows:,}</div>
                    <div class="metric-label">Rows</div>
                </div>

                <div class="metric">
                    <div class="metric-value">{total_columns:,}</div>
                    <div class="metric-label">Columns</div>
                </div>

                <div class="metric">
                    <div class="metric-value">{total_missing_cells:,}</div>
                    <div class="metric-label">Missing Cells</div>
                </div>

                <div class="metric">
                    <div class="metric-value">{missing_cells_percentage}%</div>
                    <div class="metric-label">Missing Cells %</div>
                </div>

                <div class="metric">
                    <div class="metric-value">{duplicate_rows:,}</div>
                    <div class="metric-label">Duplicate Rows</div>
                </div>

                <div class="metric">
                    <div class="metric-value">{len(numeric_cols)}</div>
                    <div class="metric-label">Numeric Columns</div>
                </div>

                <div class="metric">
                    <div class="metric-value">{len(categorical_cols)}</div>
                    <div class="metric-label">Categorical Columns</div>
                </div>

                <div class="metric">
                    <div class="metric-value">{outlier_columns_count}</div>
                    <div class="metric-label">Outlier Columns</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Smart Recommendations</h2>
            {smart_recommendations.to_html(index=False)}
        </div>

        <div class="card">
            <h2>Column Information</h2>
            {dtype_report.to_html(index=False)}
        </div>

        <div class="card">
            <h2>Missing Values Report</h2>
            {missing_report.to_html(index=False)}
        </div>

        <div class="card">
            <h2>Unique Values Report</h2>
            {unique_report.to_html(index=False)}
        </div>

        <div class="card">
            <h2>Outlier Report</h2>
            {outlier_html}
        </div>

        <div class="card">
            <h2>Structural Checks</h2>
            <p><strong>Possible ID columns:</strong> {", ".join(possible_id_cols) if possible_id_cols else "None detected"}</p>
            <p><strong>High-cardinality columns:</strong> {", ".join(high_cardinality_cols) if high_cardinality_cols else "None detected"}</p>
            <p><strong>Constant columns:</strong> {", ".join(constant_cols) if constant_cols else "None detected"}</p>
            <p><strong>Date-like columns:</strong> {", ".join(date_like_cols) if date_like_cols else "None detected"}</p>
        </div>
    </body>
    </html>
    """

    return html


def make_download_button(dataframe, file_name, label):
    csv = dataframe.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=label,
        data=csv,
        file_name=file_name,
        mime="text/csv",
        use_container_width=True
    )


def display_hero():
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">Smart EDA Dashboard</div>
            <div class="hero-subtitle">
                Upload your CSV or Excel file and explore data quality, cleaning, imputation, modeling, outliers, and correlations interactively.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# Header
# ============================================================
display_hero()


# ============================================================
# Sidebar
# ============================================================
st.sidebar.markdown("## 📊 Smart EDA")
st.sidebar.markdown("Upload a CSV or Excel file to start your analysis.")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

st.sidebar.markdown("---")

plot_template = st.sidebar.selectbox(
    "Plot style",
    ["plotly_white", "plotly", "ggplot2", "simple_white"]
)

high_cardinality_threshold = st.sidebar.slider(
    "High-cardinality threshold",
    min_value=10,
    max_value=200,
    value=50,
    step=10
)

strong_corr_threshold = st.sidebar.slider(
    "Strong correlation threshold",
    min_value=0.30,
    max_value=0.90,
    value=0.50,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("### What this app does")
st.sidebar.markdown(
    """
    - Dataset overview  
    - Data quality checks  
    - Smart recommendations  
    - Data cleaning  
    - Basic and advanced imputation  
    - Multiple imputation  
    - Statistical modeling  
    - Outlier handling  
    - Interactive visualizations  
    - Full HTML report  
    """
)


# ============================================================
# Main application
# ============================================================
if uploaded_file is not None:

    MAX_FILE_SIZE_MB = 100
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File size ({file_size_mb:.1f} MB) exceeds the {MAX_FILE_SIZE_MB} MB limit.")
        st.stop()

    try:
        sheet_names = get_excel_sheet_names(uploaded_file)
        selected_sheet = None
        if sheet_names:
            selected_sheet = st.sidebar.selectbox("Choose Excel sheet", sheet_names)

        df = load_dataframe(uploaded_file.getvalue(), uploaded_file.name.lower(), selected_sheet)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        total_rows = df.shape[0]
        total_columns = df.shape[1]
        total_missing_cells = int(df.isnull().sum().sum())
        total_cells = total_rows * total_columns

        missing_cells_percentage = (
            round((total_missing_cells / total_cells) * 100, 2)
            if total_cells > 0
            else 0
        )

        duplicate_rows = int(df.duplicated().sum())

        possible_id_cols = detect_possible_id_columns(df)
        high_cardinality_cols = detect_high_cardinality_columns(
            df,
            threshold=high_cardinality_threshold
        )
        constant_cols = detect_constant_columns(df)
        date_like_cols = detect_date_like_columns(df)

        outlier_report = calculate_outliers_iqr(df, numeric_cols)
        missing_report = create_missing_report(df)
        dtype_report = create_dtype_report(df)
        unique_report = create_unique_report(df)

        smart_recommendations = generate_smart_recommendations(
            df=df,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            missing_report=missing_report,
            duplicate_rows=duplicate_rows,
            possible_id_cols=possible_id_cols,
            high_cardinality_cols=high_cardinality_cols,
            constant_cols=constant_cols,
            date_like_cols=date_like_cols,
            outlier_report=outlier_report
        )

        outlier_columns_count = 0

        if not outlier_report.empty:
            outlier_columns_count = int(
                (outlier_report["Outliers Count"] > 0).sum()
            )

        st.sidebar.markdown("### Uploaded file")
        st.sidebar.write(f"**File name:** {uploaded_file.name}")

        if selected_sheet is not None:
            st.sidebar.write(f"**Excel sheet:** {selected_sheet}")

        st.sidebar.write(f"**Rows:** {total_rows}")
        st.sidebar.write(f"**Columns:** {total_columns}")
        st.sidebar.write(f"**Numeric:** {len(numeric_cols)}")
        st.sidebar.write(f"**Categorical:** {len(categorical_cols)}")

        st.success("File uploaded successfully. EDA report is ready.")

        # ====================================================
        # Top dashboard metrics
        # ====================================================
        st.markdown("## Dataset Snapshot")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{total_rows:,}")
        c2.metric("Columns", f"{total_columns:,}")
        c3.metric("Missing Cells", f"{total_missing_cells:,}", f"{missing_cells_percentage}%")
        c4.metric("Duplicate Rows", f"{duplicate_rows:,}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Numeric Columns", len(numeric_cols))
        c6.metric("Categorical Columns", len(categorical_cols))
        c7.metric("Possible ID Columns", len(possible_id_cols))
        c8.metric("Outlier Columns", outlier_columns_count)

        st.markdown("---")

        df_cleaned = df.copy()

        # ====================================================
        # Tabs
        # ====================================================
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "🏠 Overview",
            "🧪 Data Quality",
            "🧹 Data Cleaning",
            "📈 Visual Analysis",
            "🤖 Model",
            "🔥 Correlation",
            "📥 Reports",
            "🧾 Raw Data"
        ])

        # ====================================================
        # Tab 1: Overview
        # ====================================================
        with tab1:
            st.markdown("## Overview")

            st.markdown("### Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            st.markdown("### Column Information")
            st.dataframe(dtype_report, use_container_width=True)

            st.markdown("### Summary Statistics")
            st.dataframe(df.describe(include="all").T, use_container_width=True)

            st.markdown("### Column Type Distribution")

            type_counts = dtype_report["Data Type"].value_counts().reset_index()
            type_counts.columns = ["Data Type", "Count"]

            fig_types = px.bar(
                type_counts,
                x="Data Type",
                y="Count",
                text="Count",
                title="Column Data Types",
                template=plot_template
            )

            fig_types.update_layout(
                xaxis_title="Data Type",
                yaxis_title="Number of Columns"
            )

            st.plotly_chart(fig_types, use_container_width=True)

        # ====================================================
        # Tab 2: Data Quality
        # ====================================================
        with tab2:
            st.markdown("## Data Quality Report")

            st.markdown("### Smart Recommendations")

            high_priority_count = smart_recommendations[
                smart_recommendations["Priority"] == "High"
            ].shape[0]

            medium_priority_count = smart_recommendations[
                smart_recommendations["Priority"] == "Medium"
            ].shape[0]

            low_priority_count = smart_recommendations[
                smart_recommendations["Priority"] == "Low"
            ].shape[0]

            rec1, rec2, rec3 = st.columns(3)

            rec1.metric("High Priority", high_priority_count)
            rec2.metric("Medium Priority", medium_priority_count)
            rec3.metric("Low Priority", low_priority_count)

            recommendation_counts = (
                smart_recommendations["Priority"]
                .value_counts()
                .reset_index()
            )

            recommendation_counts.columns = ["Priority", "Count"]

            fig_recommendations = px.bar(
                recommendation_counts,
                x="Priority",
                y="Count",
                text="Count",
                title="Recommendations by Priority",
                template=plot_template,
                color="Priority",
                color_discrete_map={
                    "High": "#DC2626",
                    "Medium": "#F59E0B",
                    "Low": "#16A34A"
                }
            )

            fig_recommendations.update_layout(
                xaxis_title="Priority",
                yaxis_title="Number of Recommendations"
            )

            st.plotly_chart(fig_recommendations, use_container_width=True)

            st.dataframe(
                smart_recommendations,
                use_container_width=True
            )

            st.markdown("---")

            max_missing = missing_report["Missing Percentage"].max()
            quality_status = get_data_quality_status(max_missing)

            q1, q2, q3 = st.columns(3)

            q1.metric("Overall Missing Cells", f"{missing_cells_percentage}%")
            q2.metric("Max Column Missing", f"{max_missing}%")
            q3.metric("Quality Status", quality_status)

            st.markdown("### Missing Values")

            st.dataframe(
                missing_report,
                use_container_width=True
            )

            missing_nonzero = missing_report[
                missing_report["Missing Count"] > 0
            ]

            if not missing_nonzero.empty:
                fig_missing = px.bar(
                    missing_nonzero,
                    x="Column",
                    y="Missing Percentage",
                    text="Missing Percentage",
                    title="Missing Values Percentage by Column",
                    template=plot_template,
                    color="Missing Percentage",
                    color_continuous_scale="Reds"
                )

                fig_missing.update_layout(
                    xaxis_title="Column",
                    yaxis_title="Missing Percentage (%)"
                )

                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("No missing values found.")

            st.markdown("### Duplicate Rows")

            if duplicate_rows > 0:
                st.warning(f"There are {duplicate_rows} duplicate rows.")
                st.dataframe(df[df.duplicated()].head(20), use_container_width=True)
            else:
                st.success("No duplicate rows found.")

            st.markdown("### Structural Checks")

            s1, s2 = st.columns(2)

            with s1:
                st.markdown("#### Constant Columns")
                if constant_cols:
                    st.warning("These columns have only one unique value:")
                    st.write(constant_cols)
                else:
                    st.success("No constant columns found.")

                st.markdown("#### Possible ID Columns")
                if possible_id_cols:
                    st.warning("These columns may be identifiers:")
                    st.write(possible_id_cols)
                else:
                    st.success("No possible ID columns detected.")

            with s2:
                st.markdown("#### High-cardinality Columns")
                if high_cardinality_cols:
                    st.warning("These categorical columns have many unique values:")
                    st.write(high_cardinality_cols)
                else:
                    st.success("No high-cardinality categorical columns detected.")

                st.markdown("#### Date-like Columns")
                if date_like_cols:
                    st.info("These columns look like dates:")
                    st.write(date_like_cols)
                else:
                    st.info("No date-like columns detected.")

            st.markdown("### Unique Values Report")
            st.dataframe(unique_report, use_container_width=True)

            fig_unique = px.bar(
                unique_report.head(30),
                x="Column",
                y="Unique Values",
                text="Unique Values",
                title="Top Columns by Number of Unique Values",
                template=plot_template,
                color="Unique Values",
                color_continuous_scale="Blues"
            )

            st.plotly_chart(fig_unique, use_container_width=True)

            st.markdown("### Outlier Report")

            if numeric_cols:
                outlier_analysis_method = st.radio(
                    "Outlier detection method",
                    ["IQR", "Z-Score"],
                    horizontal=True,
                    key="tab2_outlier_method"
                )

                if outlier_analysis_method == "Z-Score":
                    z_display_threshold = st.slider(
                        "Z-Score threshold",
                        min_value=1.5,
                        max_value=5.0,
                        value=3.0,
                        step=0.5,
                        key="tab2_zscore_threshold"
                    )
                    active_outlier_report = calculate_outliers_zscore(
                        df, tuple(numeric_cols), threshold=z_display_threshold
                    )
                else:
                    active_outlier_report = outlier_report

                if not active_outlier_report.empty:
                    st.dataframe(active_outlier_report, use_container_width=True)

                    outlier_nonzero = active_outlier_report[
                        active_outlier_report["Outliers Count"] > 0
                    ]

                    if not outlier_nonzero.empty:
                        fig_outliers = px.bar(
                            outlier_nonzero,
                            x="Column",
                            y="Outliers Percentage",
                            text="Outliers Percentage",
                            title=f"Outliers Percentage by Numeric Column ({outlier_analysis_method})",
                            template=plot_template,
                            color="Outliers Percentage",
                            color_continuous_scale="Oranges"
                        )
                        fig_outliers.update_layout(
                            xaxis_title="Column",
                            yaxis_title="Outliers Percentage (%)"
                        )
                        st.plotly_chart(fig_outliers, use_container_width=True)
                    else:
                        st.success(f"No outliers detected using {outlier_analysis_method}.")
                else:
                    st.info("No numeric columns available for outlier detection.")
            else:
                st.info("No numeric columns available for outlier detection.")

        # ====================================================
        # Tab 3: Data Cleaning
        # ====================================================
        with tab3:
            st.markdown("## Data Cleaning and Imputation")

            st.warning(
                "Cleaning and imputation are applied to a copy of the data only. "
                "The original uploaded dataset is not changed."
            )

            before_missing = int(df.isnull().sum().sum())
            before_duplicates = int(df.duplicated().sum())

            st.markdown("### Before Cleaning")

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Rows", f"{df.shape[0]:,}")
            b2.metric("Columns", f"{df.shape[1]:,}")
            b3.metric("Missing Cells", f"{before_missing:,}")
            b4.metric("Duplicate Rows", f"{before_duplicates:,}")

            st.markdown("---")

            st.markdown("### Basic Cleaning Options")

            c1, c2, c3 = st.columns(3)

            with c1:
                remove_duplicates = st.checkbox(
                    "Remove duplicate rows",
                    value=True
                )

            with c2:
                remove_constant_cols = st.checkbox(
                    "Remove constant columns",
                    value=True
                )

            with c3:
                remove_id_cols = st.checkbox(
                    "Remove possible ID columns",
                    value=False
                )

            drop_missing_threshold = st.slider(
                "Drop columns with missing values above this percentage",
                min_value=0,
                max_value=100,
                value=100,
                step=5
            )

            st.caption(
                "Example: If you choose 50%, columns with more than 50% missing values will be removed. "
                "Choose 100% to disable this option."
            )

            st.markdown("---")

            st.markdown("### Missing Value Imputation")

            all_group_options = ["None"] + categorical_cols

            group_col = st.selectbox(
                "Optional grouping column for group-based imputation",
                all_group_options
            )

            i1, i2 = st.columns(2)

            with i1:
                numeric_imputation_method = st.selectbox(
                    "Numeric imputation method",
                    [
                        "Do nothing",
                        "Mean",
                        "Median",
                        "Zero",
                        "Group mean",
                        "Group median",
                        "KNN",
                        "Iterative",
                        "Multiple imputation"
                    ]
                )

            with i2:
                categorical_imputation_method = st.selectbox(
                    "Categorical imputation method",
                    [
                        "Do nothing",
                        "Mode",
                        "Unknown",
                        "Group mode"
                    ]
                )

            knn_neighbors = 5
            iterative_max_iter = 10
            multiple_n_datasets = 5
            multiple_max_iter = 10
            multiple_random_seed = 42

            if numeric_imputation_method == "KNN":
                knn_neighbors = st.slider(
                    "KNN number of neighbors",
                    min_value=2,
                    max_value=20,
                    value=5
                )

            if numeric_imputation_method == "Iterative":
                iterative_max_iter = st.slider(
                    "Iterative imputation max iterations",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5
                )

            if numeric_imputation_method == "Multiple imputation":
                st.info(
                    "Multiple imputation creates several plausible datasets. "
                    "In this version, it is applied to numeric columns only. "
                    "For formal statistical inference, analyze each dataset separately and pool results using Rubin's Rules."
                )

                m1, m2, m3 = st.columns(3)

                with m1:
                    multiple_n_datasets = st.slider(
                        "Number of imputed datasets",
                        min_value=2,
                        max_value=20,
                        value=5
                    )

                with m2:
                    multiple_max_iter = st.slider(
                        "Multiple imputation max iterations",
                        min_value=5,
                        max_value=50,
                        value=10,
                        step=5
                    )

                with m3:
                    multiple_random_seed = st.number_input(
                        "Random seed",
                        min_value=1,
                        value=42
                    )

            st.markdown("---")

            st.markdown("### Outlier Handling")

            outlier_method = st.selectbox(
                "Outlier handling method",
                [
                    "Do nothing",
                    "Cap using IQR",
                    "Remove rows using IQR",
                    "Cap using Z-Score",
                    "Remove rows using Z-Score"
                ]
            )

            z_score_clean_threshold = 3.0

            if outlier_method in ["Cap using IQR", "Remove rows using IQR"]:
                st.info(
                    "IQR method: values outside Q1 - 1.5×IQR and Q3 + 1.5×IQR are treated as outliers."
                )

            if outlier_method in ["Cap using Z-Score", "Remove rows using Z-Score"]:
                z_score_clean_threshold = st.slider(
                    "Z-Score threshold",
                    min_value=1.5,
                    max_value=5.0,
                    value=3.0,
                    step=0.5,
                    key="tab3_zscore_threshold"
                )
                st.info(
                    f"Z-Score method: values with |z| > {z_score_clean_threshold} are treated as outliers."
                )

            if outlier_method in ["Remove rows using IQR", "Remove rows using Z-Score"]:
                st.warning(
                    "Removing outlier rows can delete many observations, especially if there are many numeric columns. "
                    "Use this carefully."
                )

            st.markdown("---")

            with st.spinner("Applying cleaning and imputation..."):
                df_cleaned = clean_dataset(
                    df=df,
                    remove_duplicates=remove_duplicates,
                    remove_constant_cols=remove_constant_cols,
                    remove_id_cols=remove_id_cols,
                    drop_missing_threshold=drop_missing_threshold,
                    numeric_imputation_method=(
                        "Do nothing"
                        if numeric_imputation_method == "Multiple imputation"
                        else numeric_imputation_method
                    ),
                    categorical_imputation_method=categorical_imputation_method,
                    group_col=group_col,
                    outlier_method=outlier_method,
                    possible_id_cols=possible_id_cols,
                    constant_cols=constant_cols,
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    knn_neighbors=knn_neighbors,
                    iterative_max_iter=iterative_max_iter,
                    z_score_threshold=z_score_clean_threshold
                )

            after_missing = int(df_cleaned.isnull().sum().sum())
            after_duplicates = int(df_cleaned.duplicated().sum())

            st.markdown("### After Cleaning Preview")

            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Rows", f"{df_cleaned.shape[0]:,}", f"{df_cleaned.shape[0] - df.shape[0]:,}")
            a2.metric("Columns", f"{df_cleaned.shape[1]:,}", f"{df_cleaned.shape[1] - df.shape[1]:,}")
            a3.metric("Missing Cells", f"{after_missing:,}", f"{after_missing - before_missing:,}")
            a4.metric("Duplicate Rows", f"{after_duplicates:,}", f"{after_duplicates - before_duplicates:,}")

            st.dataframe(df_cleaned.head(20), use_container_width=True)

            st.markdown("### Download Cleaned Dataset")

            cleaned_csv = df_cleaned.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download cleaned dataset as CSV",
                data=cleaned_csv,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

            if numeric_imputation_method == "Multiple imputation":
                st.markdown("### Download Multiple Imputed Datasets")

                with st.spinner(f"Creating {multiple_n_datasets} imputed datasets..."):
                    multiple_zip = create_multiple_imputed_zip(
                        df=df_cleaned,
                        numeric_cols=df_cleaned.select_dtypes(include=np.number).columns.tolist(),
                        n_datasets=multiple_n_datasets,
                        max_iter=multiple_max_iter,
                        random_seed=int(multiple_random_seed)
                    )

                if multiple_zip is not None:
                    st.download_button(
                        label="Download multiple imputed datasets as ZIP",
                        data=multiple_zip,
                        file_name="multiple_imputed_datasets.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                else:
                    st.warning(
                        "Multiple imputation could not be created because there are no usable numeric columns."
                    )

        # ====================================================
        # Tab 4: Visual Analysis
        # ====================================================
        with tab4:
            st.markdown("## Visual Analysis")

            visual_type = st.radio(
                "Choose visualization type",
                [
                    "Numeric Distribution",
                    "Boxplot",
                    "Violin Plot",
                    "Scatter Plot",
                    "Categorical Bar Chart",
                    "Pie Chart"
                ],
                horizontal=True
            )

            if visual_type in ["Numeric Distribution", "Boxplot", "Violin Plot"]:
                if numeric_cols:
                    selected_numeric_col = st.selectbox(
                        "Choose numeric column",
                        numeric_cols
                    )

                    v1, v2, v3, v4 = st.columns(4)
                    v1.metric("Mean", round(df[selected_numeric_col].mean(), 2))
                    v2.metric("Median", round(df[selected_numeric_col].median(), 2))
                    v3.metric("Min", round(df[selected_numeric_col].min(), 2))
                    v4.metric("Max", round(df[selected_numeric_col].max(), 2))

                    if visual_type == "Numeric Distribution":
                        fig = px.histogram(
                            df,
                            x=selected_numeric_col,
                            nbins=30,
                            marginal="box",
                            title=f"Distribution of {selected_numeric_col}",
                            template=plot_template
                        )

                    elif visual_type == "Boxplot":
                        fig = px.box(
                            df,
                            x=selected_numeric_col,
                            points="outliers",
                            title=f"Boxplot of {selected_numeric_col}",
                            template=plot_template
                        )

                    else:
                        fig = px.violin(
                            df,
                            y=selected_numeric_col,
                            box=True,
                            points="outliers",
                            title=f"Violin Plot of {selected_numeric_col}",
                            template=plot_template
                        )

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("No numeric columns found.")

            elif visual_type == "Scatter Plot":
                if len(numeric_cols) >= 2:
                    sc1, sc2, sc3 = st.columns(3)

                    with sc1:
                        x_col = st.selectbox(
                            "X-axis",
                            numeric_cols,
                            key="scatter_x"
                        )

                    with sc2:
                        y_col = st.selectbox(
                            "Y-axis",
                            numeric_cols,
                            key="scatter_y"
                        )

                    with sc3:
                        color_col = None

                        if categorical_cols:
                            color_col = st.selectbox(
                                "Color by",
                                ["None"] + categorical_cols,
                                key="scatter_color"
                            )

                            if color_col == "None":
                                color_col = None

                    fig_scatter = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        title=f"Scatter Plot: {x_col} vs {y_col}",
                        template=plot_template
                    )

                    st.plotly_chart(fig_scatter, use_container_width=True)

                else:
                    st.info("At least two numeric columns are needed for scatter plot.")

            elif visual_type == "Categorical Bar Chart":
                if categorical_cols:
                    selected_cat_col = st.selectbox(
                        "Choose categorical column",
                        categorical_cols,
                        key="bar_cat"
                    )

                    top_n = st.slider(
                        "Top categories",
                        min_value=5,
                        max_value=50,
                        value=20,
                        key="bar_top_n"
                    )

                    value_counts = (
                        df[selected_cat_col]
                        .astype(str)
                        .value_counts()
                        .head(top_n)
                        .reset_index()
                    )

                    value_counts.columns = [selected_cat_col, "Count"]

                    fig_bar = px.bar(
                        value_counts,
                        x=selected_cat_col,
                        y="Count",
                        text="Count",
                        title=f"Top {top_n} Categories in {selected_cat_col}",
                        template=plot_template,
                        color="Count",
                        color_continuous_scale="Blues"
                    )

                    fig_bar.update_layout(
                        xaxis_title=selected_cat_col,
                        yaxis_title="Count"
                    )

                    st.plotly_chart(fig_bar, use_container_width=True)

                    st.dataframe(value_counts, use_container_width=True)

                else:
                    st.info("No categorical columns found.")

            elif visual_type == "Pie Chart":
                if categorical_cols:
                    selected_cat_col = st.selectbox(
                        "Choose categorical column",
                        categorical_cols,
                        key="pie_cat"
                    )

                    unique_count = df[selected_cat_col].nunique(dropna=True)

                    if unique_count <= 15:
                        value_counts = (
                            df[selected_cat_col]
                            .astype(str)
                            .value_counts()
                            .reset_index()
                        )

                        value_counts.columns = [selected_cat_col, "Count"]

                        fig_pie = px.pie(
                            value_counts,
                            names=selected_cat_col,
                            values="Count",
                            title=f"Pie Chart of {selected_cat_col}",
                            template=plot_template
                        )

                        st.plotly_chart(fig_pie, use_container_width=True)

                    else:
                        st.warning(
                            "This column has many categories. Pie chart is not recommended. Use bar chart instead."
                        )

                else:
                    st.info("No categorical columns found.")

        # ====================================================
        # Tab 5: Model
        # ====================================================
        with tab5:
            st.markdown("## Statistical Modeling")

            st.info(
                "This section fits statistical models using selected target and predictor variables. "
                "Rows with missing values in the selected variables will be removed before modeling."
            )

            dataset_choice = st.radio(
                "Choose dataset for modeling",
                ["Original data", "Cleaned data from Data Cleaning tab"],
                horizontal=True
            )

            if dataset_choice == "Cleaned data from Data Cleaning tab":
                try:
                    modeling_df = df_cleaned.copy()
                except NameError:
                    st.warning("Cleaned data is not available yet. Using original data instead.")
                    modeling_df = df.copy()
            else:
                modeling_df = df.copy()

            all_columns = modeling_df.columns.tolist()

            if len(all_columns) < 2:
                st.warning("You need at least two columns to build a model.")
            else:
                m1, m2 = st.columns(2)

                with m1:
                    target_col = st.selectbox(
                        "Select target / outcome variable",
                        all_columns
                    )

                with m2:
                    model_type = st.selectbox(
                        "Select model type",
                        [
                            "Linear Regression",
                            "Binary Logistic Regression",
                            "Poisson Regression",
                            "Negative Binomial Regression"
                        ]
                    )

                available_predictors = [
                    col for col in all_columns
                    if col != target_col
                ]

                predictor_cols = st.multiselect(
                    "Select predictor variables",
                    available_predictors,
                    default=available_predictors[:min(5, len(available_predictors))]
                )

                st.markdown("### Model Guidance")

                if model_type == "Linear Regression":
                    st.write(
                        "Use Linear Regression when the target variable is continuous numeric, "
                        "for example length of stay, BMI, glucose level, or cost."
                    )

                elif model_type == "Binary Logistic Regression":
                    st.write(
                        "Use Binary Logistic Regression when the target has exactly two categories, "
                        "for example Yes/No, 0/1, readmitted/not readmitted, disease/no disease."
                    )

                elif model_type == "Poisson Regression":
                    st.write(
                        "Use Poisson Regression when the target is count data, "
                        "for example number of visits, admissions, events, or medications."
                    )

                elif model_type == "Negative Binomial Regression":
                    st.write(
                        "Use Negative Binomial Regression for count data when the variance is larger than the mean "
                        "or when Poisson appears too restrictive."
                    )

                run_model = st.button(
                    "Run Model",
                    use_container_width=True
                )

                if run_model:
                    if len(predictor_cols) == 0:
                        st.error("Please select at least one predictor variable.")
                    else:
                        try:
                            model_df = prepare_model_data(
                                modeling_df,
                                target_col,
                                predictor_cols
                            )

                            if model_df.empty:
                                st.error("No data left after dropping missing values in selected variables.")
                            else:
                                if model_type in ["Poisson Regression", "Negative Binomial Regression"]:
                                    if not pd.api.types.is_numeric_dtype(model_df[target_col]):
                                        st.error("Count models require a numeric target variable.")
                                        st.stop()

                                    if not np.all(np.equal(np.mod(model_df[target_col], 1), 0)):
                                        st.warning(
                                            "The target variable has non-integer values. "
                                            "Poisson and Negative Binomial are usually intended for count outcomes."
                                        )

                                    target_mean = model_df[target_col].mean()
                                    target_var = model_df[target_col].var()

                                    st.write(f"Target mean: **{target_mean:.3f}**")
                                    st.write(f"Target variance: **{target_var:.3f}**")

                                    if target_var > target_mean * 1.5:
                                        st.warning(
                                            "The target variance is much larger than the mean. "
                                            "Negative Binomial may be more appropriate than Poisson."
                                        )

                                model_result, fitted_df, formula = run_statistical_model(
                                    model_df=model_df,
                                    target_col=target_col,
                                    predictor_cols=predictor_cols,
                                    model_type=model_type
                                )

                                st.success("Model fitted successfully.")

                                st.markdown("### Model Formula")
                                st.code(formula)

                                st.markdown("### Model Fit Summary")

                                fit1, fit2, fit3, fit4 = st.columns(4)

                                fit1.metric("Observations", int(model_result.nobs))

                                if hasattr(model_result, "rsquared"):
                                    fit2.metric("R-squared", round(model_result.rsquared, 4))
                                else:
                                    fit2.metric("AIC", round(model_result.aic, 2))

                                if hasattr(model_result, "rsquared_adj"):
                                    fit3.metric("Adj. R-squared", round(model_result.rsquared_adj, 4))
                                else:
                                    fit3.metric(
                                        "BIC",
                                        round(model_result.bic, 2) if hasattr(model_result, "bic") else "N/A"
                                    )

                                fit4.metric(
                                    "Log-Likelihood",
                                    round(model_result.llf, 2) if hasattr(model_result, "llf") else "N/A"
                                )

                                st.markdown("### Coefficients Table")

                                coef_table = make_coefficients_table(
                                    model_result,
                                    model_type
                                )

                                st.dataframe(
                                    coef_table,
                                    use_container_width=True
                                )

                                csv_coef = coef_table.to_csv(index=False).encode("utf-8")

                                st.download_button(
                                    label="Download coefficients table",
                                    data=csv_coef,
                                    file_name="model_coefficients.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )

                                st.markdown("### Full Model Summary")

                                summary_text = model_result.summary().as_text()
                                st.text(summary_text)

                                create_text_download(
                                    summary_text,
                                    "model_summary.txt",
                                    "Download full model summary"
                                )

                                st.markdown("### Predictions and Evaluation")

                                predictions = model_result.predict(fitted_df)

                                if model_type == "Linear Regression":
                                    eval_df = pd.DataFrame({
                                        "Actual": fitted_df[target_col],
                                        "Predicted": predictions
                                    })

                                    fig_pred = px.scatter(
                                        eval_df,
                                        x="Actual",
                                        y="Predicted",
                                        title="Actual vs Predicted",
                                        template=plot_template
                                    )

                                    st.plotly_chart(fig_pred, use_container_width=True)

                                    residuals = fitted_df[target_col] - predictions

                                    residual_df = pd.DataFrame({
                                        "Predicted": predictions,
                                        "Residuals": residuals
                                    })

                                    fig_resid = px.scatter(
                                        residual_df,
                                        x="Predicted",
                                        y="Residuals",
                                        title="Residuals vs Predicted",
                                        template=plot_template
                                    )

                                    fig_resid.add_hline(y=0)

                                    st.plotly_chart(fig_resid, use_container_width=True)

                                elif model_type == "Binary Logistic Regression":
                                    pred_prob = predictions
                                    pred_class = (pred_prob >= 0.5).astype(int)

                                    y_true = fitted_df[target_col]

                                    if not set(y_true.dropna().unique()).issubset({0, 1}):
                                        sorted_classes = sorted(y_true.dropna().unique())
                                        mapping = {
                                            sorted_classes[0]: 0,
                                            sorted_classes[1]: 1
                                        }
                                        y_true = y_true.map(mapping)

                                    accuracy = accuracy_score(y_true, pred_class)

                                    st.metric("Accuracy", round(accuracy, 4))

                                    cm = confusion_matrix(y_true, pred_class)

                                    cm_df = pd.DataFrame(
                                        cm,
                                        index=["Actual 0", "Actual 1"],
                                        columns=["Predicted 0", "Predicted 1"]
                                    )

                                    st.markdown("#### Confusion Matrix")
                                    st.dataframe(cm_df, use_container_width=True)

                                    fig_cm = px.imshow(
                                        cm_df,
                                        text_auto=True,
                                        aspect="auto",
                                        title="Confusion Matrix",
                                        color_continuous_scale="Blues"
                                    )

                                    st.plotly_chart(fig_cm, use_container_width=True)

                                    try:
                                        auc = roc_auc_score(y_true, pred_prob)
                                        st.metric("AUC", round(auc, 4))

                                        fpr, tpr, thresholds = roc_curve(y_true, pred_prob)

                                        roc_df = pd.DataFrame({
                                            "False Positive Rate": fpr,
                                            "True Positive Rate": tpr
                                        })

                                        fig_roc = px.line(
                                            roc_df,
                                            x="False Positive Rate",
                                            y="True Positive Rate",
                                            title="ROC Curve",
                                            template=plot_template
                                        )

                                        fig_roc.add_shape(
                                            type="line",
                                            x0=0,
                                            y0=0,
                                            x1=1,
                                            y1=1,
                                            line=dict(dash="dash")
                                        )

                                        st.plotly_chart(fig_roc, use_container_width=True)

                                    except Exception:
                                        st.info("AUC/ROC could not be calculated for this model.")

                                    report = classification_report(
                                        y_true,
                                        pred_class,
                                        output_dict=True
                                    )

                                    report_df = pd.DataFrame(report).T

                                    st.markdown("#### Classification Report")
                                    st.dataframe(report_df, use_container_width=True)

                                elif model_type in ["Poisson Regression", "Negative Binomial Regression"]:
                                    eval_df = pd.DataFrame({
                                        "Actual Count": fitted_df[target_col],
                                        "Predicted Count": predictions
                                    })

                                    fig_count = px.scatter(
                                        eval_df,
                                        x="Actual Count",
                                        y="Predicted Count",
                                        title="Actual vs Predicted Count",
                                        template=plot_template
                                    )

                                    st.plotly_chart(fig_count, use_container_width=True)

                                    st.markdown("#### Count Prediction Preview")
                                    st.dataframe(eval_df.head(30), use_container_width=True)

                        except ValueError as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(
                                "An unexpected error occurred while fitting the model. "
                                "Check that your selected variables have sufficient non-missing data "
                                "and are compatible with the chosen model type."
                            )
                            with st.expander("Error details"):
                                st.write(e)

        # ====================================================
        # Tab 6: Correlation
        # ====================================================
        with tab6:
            st.markdown("## Correlation Analysis")

            if len(numeric_cols) >= 2:

                corr_method = st.selectbox(
                    "Choose correlation method",
                    ["pearson", "spearman", "kendall"]
                )

                corr = df[numeric_cols].corr(method=corr_method)

                st.markdown("### Correlation Heatmap")

                fig_corr = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title=f"Correlation Matrix using {corr_method.capitalize()} Method"
                )

                fig_corr.update_layout(
                    height=750,
                    template=plot_template
                )

                st.plotly_chart(fig_corr, use_container_width=True)

                st.markdown("### Correlation Table")
                st.dataframe(corr, use_container_width=True)

                st.markdown("### Strong Correlations")

                corr_pairs = corr.where(
                    np.triu(np.ones(corr.shape), k=1).astype(bool)
                )

                corr_pairs = corr_pairs.stack().reset_index()
                corr_pairs.columns = ["Variable 1", "Variable 2", "Correlation"]
                corr_pairs["Absolute Correlation"] = corr_pairs["Correlation"].abs()

                corr_pairs = corr_pairs.sort_values(
                    by="Absolute Correlation",
                    ascending=False
                )

                strong_corr = corr_pairs[
                    corr_pairs["Absolute Correlation"] >= strong_corr_threshold
                ]

                if not strong_corr.empty:
                    st.dataframe(strong_corr, use_container_width=True)

                    fig_strong = px.bar(
                        strong_corr,
                        x="Absolute Correlation",
                        y="Variable 1",
                        color="Correlation",
                        orientation="h",
                        title="Strong Correlations",
                        template=plot_template,
                        color_continuous_scale="RdBu_r"
                    )

                    st.plotly_chart(fig_strong, use_container_width=True)
                else:
                    st.info(
                        f"No strong correlations found above {strong_corr_threshold}."
                    )

            else:
                st.info("At least two numeric columns are needed for correlation analysis.")

        # ====================================================
        # Tab 7: Reports
        # ====================================================
        with tab7:
            st.markdown("## Download Reports")

            html_report = generate_html_report(
                uploaded_file_name=uploaded_file.name,
                selected_sheet=selected_sheet,
                total_rows=total_rows,
                total_columns=total_columns,
                total_missing_cells=total_missing_cells,
                missing_cells_percentage=missing_cells_percentage,
                duplicate_rows=duplicate_rows,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                possible_id_cols=possible_id_cols,
                high_cardinality_cols=high_cardinality_cols,
                constant_cols=constant_cols,
                date_like_cols=date_like_cols,
                outlier_columns_count=outlier_columns_count,
                missing_report=missing_report,
                dtype_report=dtype_report,
                unique_report=unique_report,
                outlier_report=outlier_report,
                smart_recommendations=smart_recommendations
            )

            st.markdown("### Full HTML Report")

            st.download_button(
                label="Download Full EDA Report as HTML",
                data=html_report,
                file_name="full_eda_report.html",
                mime="text/html",
                use_container_width=True
            )

            st.markdown("---")

            st.markdown("### Smart Recommendations")
            make_download_button(
                smart_recommendations,
                "smart_recommendations_report.csv",
                "Download Smart Recommendations Report"
            )

            st.markdown("---")

            r1, r2, r3 = st.columns(3)

            with r1:
                st.markdown("### Column Types")
                make_download_button(
                    dtype_report,
                    "column_types_report.csv",
                    "Download Column Types Report"
                )

            with r2:
                st.markdown("### Missing Values")
                make_download_button(
                    missing_report,
                    "missing_values_report.csv",
                    "Download Missing Values Report"
                )

            with r3:
                st.markdown("### Unique Values")
                make_download_button(
                    unique_report,
                    "unique_values_report.csv",
                    "Download Unique Values Report"
                )

            r4, r5, r6 = st.columns(3)

            with r4:
                st.markdown("### Outliers")
                if not outlier_report.empty:
                    make_download_button(
                        outlier_report,
                        "outlier_report.csv",
                        "Download Outlier Report"
                    )
                else:
                    st.info("No outlier report available.")

            with r5:
                st.markdown("### Original Data")
                csv_data = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Current Data as CSV",
                    data=csv_data,
                    file_name="uploaded_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with r6:
                st.markdown("### Summary")
                summary_report = pd.DataFrame({
                    "Metric": [
                        "File Name",
                        "Excel Sheet",
                        "Rows",
                        "Columns",
                        "Missing Cells",
                        "Missing Cells Percentage",
                        "Duplicate Rows",
                        "Numeric Columns",
                        "Categorical Columns",
                        "Possible ID Columns",
                        "High-cardinality Columns",
                        "Constant Columns",
                        "Date-like Columns",
                        "Outlier Columns"
                    ],
                    "Value": [
                        uploaded_file.name,
                        selected_sheet if selected_sheet is not None else "Not applicable",
                        total_rows,
                        total_columns,
                        total_missing_cells,
                        missing_cells_percentage,
                        duplicate_rows,
                        len(numeric_cols),
                        len(categorical_cols),
                        len(possible_id_cols),
                        len(high_cardinality_cols),
                        len(constant_cols),
                        len(date_like_cols),
                        outlier_columns_count
                    ]
                })

                summary_report["Value"] = summary_report["Value"].astype(str)

                make_download_button(
                    summary_report,
                    "summary_report.csv",
                    "Download Summary Report"
                )

            st.markdown("### Summary Report Preview")
            st.dataframe(summary_report, use_container_width=True)

        # ====================================================
        # Tab 8: Raw Data
        # ====================================================
        with tab8:
            st.markdown("## Raw Data")

            rows_to_show = st.slider(
                "Number of rows to display",
                min_value=5,
                max_value=min(1000, total_rows),
                value=min(50, total_rows)
            )

            st.dataframe(
                df.head(rows_to_show),
                use_container_width=True
            )

    except Exception as e:
        st.error("There was an error reading or processing the file.")
        st.write(e)

else:
    st.info("Please upload a CSV or Excel file from the sidebar to start the EDA.")

    st.markdown("## What you will get after uploading a file")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown(
            """
            <div class="section-card">
                <h3>📊 Dataset Overview</h3>
                <p class="small-note">
                See rows, columns, data types, missing values, and summary statistics.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with p2:
        st.markdown(
            """
            <div class="section-card">
                <h3>🧹 Data Cleaning</h3>
                <p class="small-note">
                Clean duplicates, handle missing values, apply imputation, and manage outliers.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with p3:
        st.markdown(
            """
            <div class="section-card">
                <h3>🤖 Statistical Modeling</h3>
                <p class="small-note">
                Build linear, logistic, Poisson, and negative binomial regression models.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
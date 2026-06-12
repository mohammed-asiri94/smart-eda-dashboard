# ============================================================
# Smart EDA Dashboard — app.py
# Main entry point — UI routing only
# All logic lives in modules/
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ── Module imports ───────────────────────────────────────────
from modules.data_quality import (
    detect_possible_id_columns,
    detect_high_cardinality_columns,
    detect_constant_columns,
    detect_date_like_columns,
    calculate_outliers_iqr,
    calculate_outliers_zscore,
    create_missing_report,
    create_dtype_report,
    create_unique_report,
    get_data_quality_status,
    generate_smart_recommendations,
)
from modules.cleaning import (
    clean_dataset,
    create_multiple_imputed_zip,
)
from modules.models.base_models import render_base_model_tab
from modules.models.survival import render_survival_tab
from modules.models.time_series import render_time_series_tab
from modules.models.mixed_effects import render_mixed_effects_tab
from modules.models.causal import render_causal_tab
from modules.reports import make_download_csv, generate_html_report
from modules.statistical_tests import render_statistical_tests
from modules.data_profiling import render_data_profiling


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Smart EDA | Statistical Analysis Platform",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================
st.markdown(
    """
    <style>
    .main { background-color: #F8FAFC; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .hero-container {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #38BDF8 100%);
        padding: 35px; border-radius: 20px; margin-bottom: 25px;
        box-shadow: 0px 8px 25px rgba(15,23,42,.15);
    }
    .hero-title  { color: white; font-size: 42px; font-weight: 800; margin-bottom: 8px; }
    .hero-subtitle { color: #E0F2FE; font-size: 18px; }
    .section-card {
        background: white; padding: 22px; border-radius: 16px;
        border: 1px solid #E2E8F0; box-shadow: 0 4px 14px rgba(15,23,42,.05);
        margin-bottom: 20px;
    }
    .small-note { color: #64748B; font-size: 14px; }
    div[data-testid="stMetricValue"] { font-size: 26px; font-weight: 800; color: #0F172A; }
    div[data-testid="stMetricLabel"] { font-size: 15px; color: #475569; }
    section[data-testid="stSidebar"] { background-color: #0F172A; }
    section[data-testid="stSidebar"] * { color: #F8FAFC; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF; border-radius: 12px; padding: 12px 20px;
        border: 1px solid #E2E8F0; color: #0F172A; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { background-color: #2563EB; color: white; }
    div[data-testid="stFileUploader"] label  { color: #0F172A !important; font-weight: 700 !important; font-size: 16px !important; }
    div[data-testid="stFileUploader"] section { background-color: #F8FAFC !important; border: 2px dashed #2563EB !important; border-radius: 14px !important; }
    div[data-testid="stFileUploader"] button  { color: #FFFFFF !important; background-color: #2563EB !important; border: 1px solid #2563EB !important; font-weight: 700 !important; }
    div[data-testid="stFileUploader"] button:hover { background-color: #1D4ED8 !important; }
    div[data-testid="stFileUploader"] small { color: #334155 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Session state initialisation
# ============================================================
def init_session_state():
    defaults = {
        "df_cleaned": None,
        "cleaning_applied": False,
        "last_file_id": None,
        "selected_model": "Base Models  (Linear / Logistic / Poisson / Negative Binomial)",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ============================================================
# File reading helpers
# ============================================================
def get_excel_sheet_names(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.ExcelFile(BytesIO(uploaded_file.getvalue())).sheet_names
    return None


@st.cache_data
def load_dataframe(file_bytes, file_name, sheet_name=None):
    if file_name.endswith(".csv"):
        return pd.read_csv(BytesIO(file_bytes))
    elif file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)
    raise ValueError("Unsupported file type. Please upload CSV, XLSX, or XLS.")


# ============================================================
# Hero banner
# ============================================================
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">📊 Smart EDA Dashboard</div>
        <div class="hero-subtitle">
            Exploratory Data Analysis &bull; Statistical Modeling &bull; Survival Analysis
            &bull; Time Series &bull; Mixed Effects &bull; Causal Inference
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================
st.sidebar.markdown("## 📊 Smart EDA")

uploaded_file = st.file_uploader(
    "Upload your dataset (CSV or Excel)",
    type=["csv", "xlsx", "xls"],
    key="main_uploader",
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
# Color Theme
theme_choice = st.sidebar.selectbox(
    "Color Theme",
    ["Blue", "Green", "Purple", "Orange", "Red"],
    key="color_theme",
)

THEME_COLORS = {
    "Blue":   {"primary": "#2563EB", "scale": "Blues",   "template": "plotly_white"},
    "Green":  {"primary": "#16A34A", "scale": "Greens",  "template": "plotly_white"},
    "Purple": {"primary": "#7C3AED", "scale": "Purples", "template": "plotly_white"},
    "Orange": {"primary": "#EA580C", "scale": "Oranges", "template": "plotly_white"},
    "Red":    {"primary": "#DC2626", "scale": "Reds",    "template": "plotly_white"},
}

theme = THEME_COLORS[theme_choice]
plot_template  = theme["template"]
primary_color  = theme["primary"]
color_scale    = theme["scale"]

# تطبيق اللون على CSS
st.markdown(
    f"""
    <style>
    .stTabs [aria-selected="true"] {{
        background-color: {primary_color} !important;
        color: white !important;
    }}
    div[data-testid="stFileUploader"] button {{
        background-color: {primary_color} !important;
        border-color: {primary_color} !important;
    }}
    .hero-container {{
        background: linear-gradient(135deg, #0F172A 0%, {primary_color} 100%) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)
high_cardinality_threshold = st.sidebar.slider(
    "High-cardinality threshold", 10, 200, 50, 10
)
strong_corr_threshold = st.sidebar.slider(
    "Strong correlation threshold", 0.30, 0.90, 0.50, 0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("### What this app does")
st.sidebar.markdown(
    """
- Dataset overview & summary stats
- Data quality checks & recommendations
- Data cleaning & imputation
- Outlier detection & handling
- Interactive visualizations
- Correlation analysis
- Statistical modeling:
  - Linear / Logistic / Poisson / NB
  - Survival (KM + Cox PH)
  - Time Series + ITS *(coming soon)*
  - Mixed Effects *(coming soon)*
  - Causal Inference *(coming soon)*
- Full HTML + CSV reports
    """
)


# ============================================================
# Main app
# ============================================================
if uploaded_file is not None:

    if uploaded_file.size / (1024 * 1024) > 100:
        st.error("File exceeds 100 MB limit.")
        st.stop()

    file_id = (uploaded_file.name, uploaded_file.size)
    if st.session_state["last_file_id"] != file_id:
        st.session_state["df_cleaned"] = None
        st.session_state["cleaning_applied"] = False
        st.session_state["last_file_id"] = file_id

    try:
        sheet_names = get_excel_sheet_names(uploaded_file)
        selected_sheet = None
        if sheet_names:
            selected_sheet = st.sidebar.selectbox("Choose Excel sheet", sheet_names)

        df = load_dataframe(
            uploaded_file.getvalue(), uploaded_file.name.lower(), selected_sheet
        )

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        total_rows         = df.shape[0]
        total_columns      = df.shape[1]
        total_missing_cells = int(df.isnull().sum().sum())
        total_cells        = total_rows * total_columns
        missing_cells_pct  = (
            round((total_missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0
        )
        duplicate_rows = int(df.duplicated().sum())

        possible_id_cols      = detect_possible_id_columns(df)
        high_cardinality_cols = detect_high_cardinality_columns(df, high_cardinality_threshold)
        constant_cols         = detect_constant_columns(df)
        date_like_cols        = detect_date_like_columns(df)
        outlier_report        = calculate_outliers_iqr(df, tuple(numeric_cols))
        missing_report        = create_missing_report(df)
        dtype_report          = create_dtype_report(df)
        unique_report         = create_unique_report(df)
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
        outlier_cols_count = int(
            (outlier_report["Outliers Count"] > 0).sum()
        ) if not outlier_report.empty else 0

        st.sidebar.markdown("### Uploaded file")
        st.sidebar.write(f"**Name:** {uploaded_file.name}")
        if selected_sheet:
            st.sidebar.write(f"**Sheet:** {selected_sheet}")
        st.sidebar.write(f"**Rows:** {total_rows:,}")
        st.sidebar.write(f"**Columns:** {total_columns:,}")
        st.sidebar.write(f"**Numeric:** {len(numeric_cols)}")
        st.sidebar.write(f"**Categorical:** {len(categorical_cols)}")

        st.success("✅ File loaded. EDA report is ready.")

        st.markdown("## Dataset Snapshot")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{total_rows:,}")
        c2.metric("Columns", f"{total_columns:,}")
        c3.metric("Missing Cells", f"{total_missing_cells:,}", f"{missing_cells_pct}%")
        c4.metric("Duplicate Rows", f"{duplicate_rows:,}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Numeric Columns", len(numeric_cols))
        c6.metric("Categorical Columns", len(categorical_cols))
        c7.metric("Possible ID Columns", len(possible_id_cols))
        c8.metric("Outlier Columns", outlier_cols_count)

        st.markdown("---")

        if st.session_state["df_cleaned"] is None:
            st.session_state["df_cleaned"] = df.copy()
        df_cleaned = st.session_state["df_cleaned"]

        # ============================================================
        # Stable navigation
        # Replaces st.tabs because tabs can reset to Overview after rerun.
        # ============================================================
        TAB_OPTIONS = [
            "🏠 Overview",
            "🧪 Data Quality",
            "🧹 Data Cleaning",
            "📈 Visual Analysis",
            "🤖 Models",
            "🔥 Correlation",
            "📥 Reports",
            "🧾 Raw Data",
        ]

        active_tab = st.radio(
            "Navigation",
            TAB_OPTIONS,
            horizontal=True,
            key="active_tab",
            label_visibility="collapsed",
        )

        st.markdown("---")

        if active_tab == TAB_OPTIONS[0]:
            st.markdown("## Overview")
            st.markdown("### Data Preview (first 10 rows)")
            st.dataframe(df.head(10), use_container_width=True)

            st.markdown("### Column Information")
            st.dataframe(dtype_report, use_container_width=True)

            st.markdown("### Summary Statistics")
            st.dataframe(df.describe(include="all").T, use_container_width=True)

            st.markdown("---")
            st.markdown("### Data Profiling")
            render_data_profiling(df, plot_template)

            st.markdown("### Column Type Distribution")
            type_counts = dtype_report["Data Type"].value_counts().reset_index()
            type_counts.columns = ["Data Type", "Count"]
            fig = px.bar(
                type_counts, x="Data Type", y="Count", text="Count",
                title="Column Data Types", template=plot_template,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ════════════════════════════════════════════════════
        # Tab 2 — Data Quality
        # ════════════════════════════════════════════════════
        elif active_tab == TAB_OPTIONS[1]:
            st.markdown("## Data Quality Report")

            st.markdown("### Smart Recommendations")
            high_n   = (smart_recommendations["Priority"] == "High").sum()
            medium_n = (smart_recommendations["Priority"] == "Medium").sum()
            low_n    = (smart_recommendations["Priority"] == "Low").sum()
            r1, r2, r3 = st.columns(3)
            r1.metric("🔴 High Priority", high_n)
            r2.metric("🟡 Medium Priority", medium_n)
            r3.metric("🟢 Low Priority", low_n)

            rec_counts = smart_recommendations["Priority"].value_counts().reset_index()
            rec_counts.columns = ["Priority", "Count"]
            fig_rec = px.bar(
                rec_counts, x="Priority", y="Count", text="Count",
                title="Recommendations by Priority", template=plot_template,
                color="Priority",
                color_discrete_map={"High": "#DC2626", "Medium": "#F59E0B", "Low": "#16A34A"},
            )
            st.plotly_chart(fig_rec, use_container_width=True)
            st.dataframe(smart_recommendations, use_container_width=True)

            st.markdown("---")
            max_missing = missing_report["Missing Percentage"].max()
            q1, q2, q3 = st.columns(3)
            q1.metric("Overall Missing %", f"{missing_cells_pct}%")
            q2.metric("Max Column Missing", f"{max_missing}%")
            q3.metric("Quality Status", get_data_quality_status(max_missing))

            st.markdown("### Missing Values")
            st.dataframe(missing_report, use_container_width=True)
            nonzero_missing = missing_report[missing_report["Missing Count"] > 0]
            if not nonzero_missing.empty:
                fig_m = px.bar(
                    nonzero_missing, x="Column", y="Missing Percentage",
                    text="Missing Percentage", title="Missing Values by Column",
                    template=plot_template, color="Missing Percentage",
                    color_continuous_scale="Reds",
                )
                st.plotly_chart(fig_m, use_container_width=True)
            else:
                st.success("No missing values found.")

            st.markdown("### Duplicate Rows")
            if duplicate_rows > 0:
                st.warning(f"{duplicate_rows} duplicate rows found.")
                st.dataframe(df[df.duplicated()].head(20), use_container_width=True)
            else:
                st.success("No duplicate rows found.")

            st.markdown("### Structural Checks")
            s1, s2 = st.columns(2)
            with s1:
                st.markdown("#### Constant Columns")
                if constant_cols:
                    st.warning("Only one unique value:"); st.write(constant_cols)
                else:
                    st.success("None found.")
                st.markdown("#### Possible ID Columns")
                if possible_id_cols:
                    st.warning("May be identifiers:"); st.write(possible_id_cols)
                else:
                    st.success("None detected.")
            with s2:
                st.markdown("#### High-Cardinality Columns")
                if high_cardinality_cols:
                    st.warning("Many unique values:"); st.write(high_cardinality_cols)
                else:
                    st.success("None detected.")
                st.markdown("#### Date-like Columns")
                if date_like_cols:
                    st.info("Look like dates:"); st.write(date_like_cols)
                else:
                    st.info("None detected.")

            st.markdown("### Unique Values Report")
            st.dataframe(unique_report, use_container_width=True)
            fig_u = px.bar(
                unique_report.head(30), x="Column", y="Unique Values",
                text="Unique Values", title="Top Columns by Unique Values",
                template=plot_template, color="Unique Values",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig_u, use_container_width=True)

            st.markdown("### Outlier Report")
            if numeric_cols:
                method_choice = st.radio(
                    "Outlier detection method", ["IQR", "Z-Score"],
                    horizontal=True, key="dq_outlier_method",
                )
                if method_choice == "Z-Score":
                    z_thresh = st.slider("Z-Score threshold", 1.5, 5.0, 3.0, 0.5, key="dq_z")
                    active_outlier = calculate_outliers_zscore(
                        df, tuple(numeric_cols), threshold=z_thresh
                    )
                else:
                    active_outlier = outlier_report

                if not active_outlier.empty:
                    st.dataframe(active_outlier, use_container_width=True)
                    nz = active_outlier[active_outlier["Outliers Count"] > 0]
                    if not nz.empty:
                        fig_o = px.bar(
                            nz, x="Column", y="Outliers Percentage",
                            text="Outliers Percentage",
                            title=f"Outliers % ({method_choice})",
                            template=plot_template, color="Outliers Percentage",
                            color_continuous_scale="Oranges",
                        )
                        st.plotly_chart(fig_o, use_container_width=True)
                    else:
                        st.success(f"No outliers detected via {method_choice}.")
            else:
                st.info("No numeric columns for outlier detection.")

            st.markdown("---")
            render_statistical_tests(df, plot_template)

        # ════════════════════════════════════════════════════
        # Tab 3 — Data Cleaning
        # ════════════════════════════════════════════════════
        elif active_tab == TAB_OPTIONS[2]:
            st.markdown("## Data Cleaning and Imputation")
            st.warning(
                "Cleaning is applied to a **copy** of the uploaded data. "
                "The original is never modified."
            )

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Rows (before)", f"{df.shape[0]:,}")
            b2.metric("Columns (before)", f"{df.shape[1]:,}")
            b3.metric("Missing (before)", f"{int(df.isnull().sum().sum()):,}")
            b4.metric("Duplicates (before)", f"{duplicate_rows:,}")

            st.markdown("---")
            st.markdown("### Basic Cleaning")
            cl1, cl2, cl3 = st.columns(3)
            remove_dups  = cl1.checkbox("Remove duplicate rows", value=True)
            remove_const = cl2.checkbox("Remove constant columns", value=True)
            remove_ids   = cl3.checkbox("Remove possible ID columns", value=False)
            drop_miss_thresh = st.slider(
                "Drop columns with missing % above:", 0, 100, 100, 5,
                help="Set to 100 to disable.",
            )

            st.markdown("---")
            st.markdown("### Missing Value Imputation")
            group_options = ["None"] + categorical_cols
            group_col = st.selectbox("Grouping column (optional)", group_options)
            imp1, imp2 = st.columns(2)
            with imp1:
                num_imp = st.selectbox(
                    "Numeric imputation",
                    ["Do nothing", "Mean", "Median", "Zero",
                     "Group mean", "Group median", "KNN", "Iterative",
                     "Multiple imputation"],
                )
            with imp2:
                cat_imp = st.selectbox(
                    "Categorical imputation",
                    ["Do nothing", "Mode", "Unknown", "Group mode"],
                )

            knn_k = 5
            iter_n = 10
            multi_n, multi_iter, multi_seed = 5, 10, 42
            if num_imp == "KNN":
                knn_k = st.slider("KNN neighbors", 2, 20, 5)
            if num_imp == "Iterative":
                iter_n = st.slider("Max iterations", 5, 50, 10, 5)
            if num_imp == "Multiple imputation":
                st.info("Multiple imputation creates several datasets. Download as ZIP.")
                mc1, mc2, mc3 = st.columns(3)
                multi_n    = mc1.slider("# datasets", 2, 20, 5)
                multi_iter = mc2.slider("Max iterations", 5, 50, 10, 5)
                multi_seed = int(mc3.number_input("Random seed", 1, value=42))

            st.markdown("---")
            st.markdown("### Outlier Handling")
            outlier_opt = st.selectbox(
                "Outlier method",
                ["Do nothing", "Cap using IQR", "Remove rows using IQR",
                 "Cap using Z-Score", "Remove rows using Z-Score"],
            )
            z_clean = 3.0
            if "IQR" in outlier_opt:
                st.info("IQR: values outside Q1 − 1.5×IQR and Q3 + 1.5×IQR are treated as outliers.")
            if "Z-Score" in outlier_opt:
                z_clean = st.slider("Z-Score threshold", 1.5, 5.0, 3.0, 0.5, key="cl_z")
            if "Remove rows" in outlier_opt:
                st.warning("Row removal can delete many observations. Use with care.")

            st.markdown("---")
            if st.button("▶ Apply Cleaning", use_container_width=True):
                with st.spinner("Applying cleaning and imputation…"):
                    result = clean_dataset(
                        df=df,
                        remove_duplicates=remove_dups,
                        remove_constant_cols=remove_const,
                        remove_id_cols=remove_ids,
                        drop_missing_threshold=drop_miss_thresh,
                        numeric_imputation_method=(
                            "Do nothing" if num_imp == "Multiple imputation" else num_imp
                        ),
                        categorical_imputation_method=cat_imp,
                        group_col=group_col,
                        outlier_method=outlier_opt,
                        possible_id_cols=possible_id_cols,
                        constant_cols=constant_cols,
                        numeric_cols=numeric_cols,
                        categorical_cols=categorical_cols,
                        knn_neighbors=knn_k,
                        iterative_max_iter=iter_n,
                        z_score_threshold=z_clean,
                    )
                    st.session_state["df_cleaned"] = result
                    st.session_state["cleaning_applied"] = True
                    df_cleaned = result
                st.success("✅ Cleaning applied and saved for use in the Models tab.")

            if st.session_state["cleaning_applied"]:
                st.markdown("### After Cleaning")
                dc = st.session_state["df_cleaned"]
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Rows", f"{dc.shape[0]:,}", f"{dc.shape[0]-df.shape[0]:,}")
                a2.metric("Columns", f"{dc.shape[1]:,}", f"{dc.shape[1]-df.shape[1]:,}")
                a3.metric("Missing", f"{int(dc.isnull().sum().sum()):,}",
                          f"{int(dc.isnull().sum().sum())-int(df.isnull().sum().sum()):,}")
                a4.metric("Duplicates", f"{int(dc.duplicated().sum()):,}")
                st.dataframe(dc.head(20), use_container_width=True)

                st.download_button(
                    "📥 Download cleaned dataset (CSV)",
                    data=dc.to_csv(index=False).encode(),
                    file_name="cleaned_dataset.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                if num_imp == "Multiple imputation":
                    st.markdown("### Download Multiple Imputed Datasets")
                    with st.spinner(f"Creating {multi_n} imputed datasets…"):
                        zip_buf = create_multiple_imputed_zip(
                            df=dc,
                            numeric_cols=dc.select_dtypes(include=np.number).columns.tolist(),
                            n_datasets=multi_n,
                            max_iter=multi_iter,
                            random_seed=multi_seed,
                        )
                    if zip_buf:
                        st.download_button(
                            "📥 Download imputed datasets (ZIP)",
                            data=zip_buf,
                            file_name="multiple_imputed_datasets.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )
                    else:
                        st.warning("No usable numeric columns for multiple imputation.")

        # ════════════════════════════════════════════════════
        # Tab 4 — Visual Analysis
        elif active_tab == TAB_OPTIONS[3]:
            st.markdown("## Visual Analysis")
            st.caption("Interactive charts organized by analysis type.")

            numeric_cols  = df.select_dtypes(include="number").columns.tolist()
            cat_cols      = df.select_dtypes(include=["object","category","bool"]).columns.tolist()
            all_cols      = df.columns.tolist()

            # ── 4 sections ────────────────────────────────────
            viz_section = st.radio(
                "Analysis type",
                [
                    "Univariate — Single variable",
                    "Bivariate  — Two variables",
                    "Multivariate — Multiple variables",
                    "Time & Advanced",
                ],
                horizontal=True,
                key="viz_section",
            )
            st.markdown("---")

            # ════════════════════════════════════════
            # SECTION 1 — Univariate
            # ════════════════════════════════════════
            if viz_section.startswith("Univariate"):

                col_type = st.radio(
                    "Variable type",
                    ["Numeric", "Categorical"],
                    horizontal=True, key="uni_type",
                )

                if col_type == "Numeric" and numeric_cols:
                    sel = st.selectbox("Select column", numeric_cols, key="uni_num")
                    chart_type = st.radio(
                        "Chart",
                        ["Histogram + KDE", "Boxplot", "Violin", "ECDF", "Strip Plot"],
                        horizontal=True, key="uni_num_chart",
                    )

                    s = df[sel].dropna()
                    v1,v2,v3,v4 = st.columns(4)
                    v1.metric("Mean",   round(s.mean(),3))
                    v2.metric("Median", round(s.median(),3))
                    v3.metric("Std",    round(s.std(),3))
                    v4.metric("Skew",   round(s.skew(),3))

                    if chart_type == "Histogram + KDE":
                        fig = px.histogram(
                            df, x=sel, nbins=40, marginal="violin",
                            histnorm="probability density",
                            title="Distribution of " + sel,
                            template=plot_template,
                            color_discrete_sequence=[primary_color],
                        )
                        fig.update_traces(opacity=0.75)
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "Boxplot":
                        grp_box = st.selectbox(
                            "Group by (optional)", ["None"] + cat_cols, key="uni_box_grp"
                        )
                        fig = px.box(
                            df,
                            y=sel,
                            x=None if grp_box == "None" else grp_box,
                            color=None if grp_box == "None" else grp_box,
                            points="outliers",
                            title="Boxplot of " + sel,
                            template=plot_template,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "Violin":
                        grp_vio = st.selectbox(
                            "Group by (optional)", ["None"] + cat_cols, key="uni_vio_grp"
                        )
                        fig = px.violin(
                            df,
                            y=sel,
                            x=None if grp_vio == "None" else grp_vio,
                            color=None if grp_vio == "None" else grp_vio,
                            box=True, points="outliers",
                            title="Violin Plot of " + sel,
                            template=plot_template,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "ECDF":
                        fig = px.ecdf(
                            df, x=sel,
                            title="ECDF of " + sel,
                            template=plot_template,
                            color_discrete_sequence=[primary_color],
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(
                            "ECDF shows the proportion of data below each value. "
                            "Useful for understanding percentiles."
                        )

                    elif chart_type == "Strip Plot":
                        grp_str = st.selectbox(
                            "Group by (optional)", ["None"] + cat_cols, key="uni_str_grp"
                        )
                        fig = px.strip(
                            df,
                            y=sel,
                            x=None if grp_str == "None" else grp_str,
                            color=None if grp_str == "None" else grp_str,
                            title="Strip Plot of " + sel,
                            template=plot_template,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Every point is visible — good for small datasets.")

                elif col_type == "Categorical" and cat_cols:
                    sel = st.selectbox("Select column", cat_cols, key="uni_cat")
                    chart_type = st.radio(
                        "Chart",
                        ["Bar Chart", "Horizontal Bar", "Pie Chart", "Treemap", "Funnel"],
                        horizontal=True, key="uni_cat_chart",
                    )
                    top_n = st.slider("Top N categories", 5, 50, 15, key="uni_top_n")

                    vc = df[sel].astype(str).value_counts().head(top_n).reset_index()
                    vc.columns = [sel, "Count"]
                    vc["Percentage"] = (vc["Count"] / len(df) * 100).round(2)

                    if chart_type == "Bar Chart":
                        fig = px.bar(
                            vc, x=sel, y="Count", text="Count",
                            title="Top " + str(top_n) + " in " + sel,
                            template=plot_template,
                            color="Count",
                            color_continuous_scale=color_scale,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "Horizontal Bar":
                        fig = px.bar(
                            vc.sort_values("Count"),
                            x="Count", y=sel, orientation="h",
                            text="Count",
                            title="Top " + str(top_n) + " in " + sel,
                            template=plot_template,
                            color="Count",
                            color_continuous_scale=color_scale,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "Pie Chart":
                        if vc.shape[0] <= 15:
                            fig = px.pie(
                                vc, names=sel, values="Count",
                                title="Distribution of " + sel,
                                template=plot_template,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Too many categories for pie. Use Bar Chart.")

                    elif chart_type == "Treemap":
                        fig = px.treemap(
                            vc, path=[sel], values="Count",
                            title="Treemap of " + sel,
                            template=plot_template,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "Funnel":
                        fig = px.funnel(
                            vc, x="Count", y=sel,
                            title="Funnel Chart of " + sel,
                            template=plot_template,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(vc, use_container_width=True)

                else:
                    st.info("No columns of this type available.")

            # ════════════════════════════════════════
            # SECTION 2 — Bivariate
            # ════════════════════════════════════════
            elif viz_section.startswith("Bivariate"):

                biv_type = st.radio(
                    "Combination",
                    ["Numeric × Numeric", "Numeric × Categorical", "Categorical × Categorical"],
                    horizontal=True, key="biv_type",
                )

                if biv_type == "Numeric × Numeric" and len(numeric_cols) >= 2:
                    b1,b2,b3 = st.columns(3)
                    x_col  = b1.selectbox("X axis", numeric_cols, key="biv_x")
                    y_col  = b2.selectbox("Y axis", numeric_cols, key="biv_y")
                    c_col  = b3.selectbox("Color by (optional)", ["None"]+cat_cols, key="biv_c")
                    color_arg = None if c_col == "None" else c_col

                    chart_type = st.radio(
                        "Chart",
                        ["Scatter + Trendline", "Hexbin Density", "2D Histogram", "Bubble"],
                        horizontal=True, key="biv_nn_chart",
                    )

                    if chart_type == "Scatter + Trendline":
                        fig = px.scatter(
                            df, x=x_col, y=y_col, color=color_arg,
                            trendline="ols",
                            trendline_color_override="red",
                            title=x_col + " vs " + y_col,
                            template=plot_template,
                            opacity=0.6,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        # R²
                        from scipy.stats import pearsonr
                        clean = df[[x_col, y_col]].dropna()
                        if len(clean) > 2:
                            r, p = pearsonr(clean[x_col], clean[y_col])
                            c1,c2,c3 = st.columns(3)
                            c1.metric("Pearson r",  round(r, 4))
                            c2.metric("R²",         round(r**2, 4))
                            c3.metric("p-value",    round(p, 5))

                    elif chart_type == "Hexbin Density":
                        fig = px.density_heatmap(
                            df, x=x_col, y=y_col,
                            nbinsx=30, nbinsy=30,
                            marginal_x="histogram",
                            marginal_y="histogram",
                            title="Hexbin: " + x_col + " vs " + y_col,
                            template=plot_template,
                            color_continuous_scale=color_scale,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Useful when many overlapping points hide the true density.")

                    elif chart_type == "2D Histogram":
                        fig = px.density_contour(
                            df, x=x_col, y=y_col, color=color_arg,
                            title="2D Density: " + x_col + " vs " + y_col,
                            template=plot_template,
                        )
                        fig.update_traces(contours_coloring="fill")
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "Bubble":
                        if len(numeric_cols) >= 3:
                            size_col = st.selectbox(
                                "Bubble size", numeric_cols, key="biv_bubble_size"
                            )
                            fig = px.scatter(
                                df, x=x_col, y=y_col,
                                size=size_col, color=color_arg,
                                title="Bubble: " + x_col + " vs " + y_col,
                                template=plot_template,
                                size_max=40, opacity=0.6,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Need at least 3 numeric columns for bubble chart.")

                elif biv_type == "Numeric × Categorical":
                    if not numeric_cols or not cat_cols:
                        st.info("Need at least one numeric and one categorical column.")
                    else:
                        bn1,bn2 = st.columns(2)
                        num_col = bn1.selectbox("Numeric", numeric_cols, key="biv_nc_num")
                        cat_col = bn2.selectbox("Categorical", cat_cols, key="biv_nc_cat")

                        chart_type = st.radio(
                            "Chart",
                            ["Grouped Boxplot", "Grouped Violin",
                             "Strip Plot", "Mean ± SD Bar"],
                            horizontal=True, key="biv_nc_chart",
                        )

                        if chart_type == "Grouped Boxplot":
                            fig = px.box(
                                df, x=cat_col, y=num_col,
                                color=cat_col, points="outliers",
                                title=num_col + " by " + cat_col,
                                template=plot_template,
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        elif chart_type == "Grouped Violin":
                            fig = px.violin(
                                df, x=cat_col, y=num_col,
                                color=cat_col, box=True, points="outliers",
                                title=num_col + " by " + cat_col,
                                template=plot_template,
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        elif chart_type == "Strip Plot":
                            fig = px.strip(
                                df, x=cat_col, y=num_col, color=cat_col,
                                title=num_col + " by " + cat_col,
                                template=plot_template,
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        elif chart_type == "Mean +/- SD Bar":
                            grp = df.groupby(cat_col)[num_col].agg(
                                Mean="mean", SD="std", N="count"
                            ).reset_index()
                            grp["SE"] = grp["SD"] / grp["N"].pow(0.5)
                            fig = px.bar(
                                grp, x=cat_col, y="Mean",
                                error_y="SD",
                                color=cat_col,
                                title="Mean +/- SD: " + num_col + " by " + cat_col,
                                template=plot_template,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(grp.round(3), use_container_width=True)

                elif biv_type == "Categorical × Categorical":
                    if len(cat_cols) < 2:
                        st.info("Need at least 2 categorical columns.")
                    else:
                        cc1,cc2 = st.columns(2)
                        cat1 = cc1.selectbox("Variable 1", cat_cols, key="biv_cc1")
                        cat2 = cc2.selectbox("Variable 2", cat_cols, key="biv_cc2")

                        if cat1 != cat2:
                            ct = pd.crosstab(df[cat1], df[cat2])
                            chart_type = st.radio(
                                "Chart",
                                ["Heatmap", "Grouped Bar", "Stacked Bar"],
                                horizontal=True, key="biv_cc_chart",
                            )

                            if chart_type == "Heatmap":
                                fig = px.imshow(
                                    ct, text_auto=True, aspect="auto",
                                    color_continuous_scale=color_scale,
                                    title=cat1 + " vs " + cat2,
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            elif chart_type == "Grouped Bar":
                                ct_long = ct.reset_index().melt(id_vars=cat1)
                                fig = px.bar(
                                    ct_long, x=cat1, y="value",
                                    color=cat2, barmode="group",
                                    title=cat1 + " vs " + cat2,
                                    template=plot_template,
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            elif chart_type == "Stacked Bar":
                                ct_long = ct.reset_index().melt(id_vars=cat1)
                                fig = px.bar(
                                    ct_long, x=cat1, y="value",
                                    color=cat2, barmode="stack",
                                    title=cat1 + " vs " + cat2,
                                    template=plot_template,
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            with st.expander("Contingency Table"):
                                st.dataframe(ct, use_container_width=True)

            # ════════════════════════════════════════
            # SECTION 3 — Multivariate
            # ════════════════════════════════════════
            elif viz_section.startswith("Multivariate"):

                mv_chart = st.radio(
                    "Chart type",
                    ["Pair Plot", "Parallel Coordinates",
                     "Scatter Matrix", "Radar Chart", "3D Scatter"],
                    horizontal=True, key="mv_chart",
                )

                if mv_chart == "Pair Plot":
                    if len(numeric_cols) < 2:
                        st.info("Need at least 2 numeric columns.")
                    else:
                        max_cols = min(6, len(numeric_cols))
                        sel_cols = st.multiselect(
                            "Select columns (max 6)",
                            numeric_cols,
                            default=numeric_cols[:min(4, len(numeric_cols))],
                            key="mv_pair_cols",
                        )
                        color_grp = st.selectbox(
                            "Color by (optional)", ["None"]+cat_cols, key="mv_pair_color"
                        )
                        if sel_cols:
                            fig = px.scatter_matrix(
                                df,
                                dimensions=sel_cols[:max_cols],
                                color=None if color_grp == "None" else color_grp,
                                title="Pair Plot",
                                template=plot_template,
                                opacity=0.5,
                            )
                            fig.update_traces(diagonal_visible=True)
                            fig.update_layout(height=700)
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(
                                "Diagonal = distribution of each variable. "
                                "Off-diagonal = scatter between pairs."
                            )

                elif mv_chart == "Parallel Coordinates":
                    if len(numeric_cols) < 2:
                        st.info("Need at least 2 numeric columns.")
                    else:
                        sel_cols = st.multiselect(
                            "Select numeric columns",
                            numeric_cols,
                            default=numeric_cols[:min(5, len(numeric_cols))],
                            key="mv_pc_cols",
                        )
                        color_col_pc = st.selectbox(
                            "Color by", numeric_cols, key="mv_pc_color"
                        )
                        if sel_cols:
                            fig = px.parallel_coordinates(
                                df.dropna(subset=sel_cols),
                                dimensions=sel_cols,
                                color=color_col_pc,
                                color_continuous_scale=color_scale,
                                title="Parallel Coordinates",
                                template=plot_template,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(
                                "Each line = one observation. "
                                "Drag axes to reorder and filter ranges."
                            )

                elif mv_chart == "Scatter Matrix":
                    sel_cols = st.multiselect(
                        "Select columns",
                        numeric_cols,
                        default=numeric_cols[:min(4, len(numeric_cols))],
                        key="mv_sm_cols",
                    )
                    if sel_cols and len(sel_cols) >= 2:
                        fig = px.scatter_matrix(
                            df, dimensions=sel_cols,
                            title="Scatter Matrix",
                            template=plot_template,
                            opacity=0.5,
                        )
                        fig.update_layout(height=650)
                        st.plotly_chart(fig, use_container_width=True)

                elif mv_chart == "Radar Chart":
                    if len(numeric_cols) >= 3 and cat_cols:
                        radar_cat = st.selectbox(
                            "Group variable", cat_cols, key="mv_radar_cat"
                        )
                        radar_cols = st.multiselect(
                            "Numeric variables (3-8)",
                            numeric_cols,
                            default=numeric_cols[:min(5, len(numeric_cols))],
                            key="mv_radar_cols",
                        )
                        if radar_cols and len(radar_cols) >= 3:
                            grp_means = df.groupby(radar_cat)[radar_cols].mean()
                            # normalize
                            grp_norm = (grp_means - grp_means.min()) / (
                                grp_means.max() - grp_means.min() + 1e-9
                            )
                            fig = go.Figure()
                            for grp_name in grp_norm.index:
                                vals = list(grp_norm.loc[grp_name]) + [grp_norm.loc[grp_name, radar_cols[0]]]
                                fig.add_trace(go.Scatterpolar(
                                    r=vals,
                                    theta=radar_cols + [radar_cols[0]],
                                    fill="toself",
                                    name=str(grp_name),
                                ))
                            fig.update_layout(
                                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                                title="Radar Chart by " + radar_cat,
                                template=plot_template,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("Values normalized 0-1 for comparison across groups.")
                    else:
                        st.info("Need at least 3 numeric columns and 1 categorical column.")

                elif mv_chart == "3D Scatter":
                    if len(numeric_cols) >= 3:
                        d1,d2,d3 = st.columns(3)
                        x3 = d1.selectbox("X", numeric_cols, key="mv_3dx")
                        y3 = d2.selectbox("Y", numeric_cols, key="mv_3dy")
                        z3 = d3.selectbox("Z", numeric_cols, key="mv_3dz")
                        c3 = st.selectbox("Color", ["None"]+cat_cols+numeric_cols, key="mv_3dc")
                        fig = px.scatter_3d(
                            df.dropna(subset=[x3,y3,z3]),
                            x=x3, y=y3, z=z3,
                            color=None if c3=="None" else c3,
                            title="3D Scatter: " + x3 + " / " + y3 + " / " + z3,
                            template=plot_template,
                            opacity=0.6,
                        )
                        fig.update_layout(height=650)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Need at least 3 numeric columns.")

            # ════════════════════════════════════════
            # SECTION 4 — Time & Advanced
            # ════════════════════════════════════════
            elif viz_section.startswith("Time"):

                adv_chart = st.radio(
                    "Chart type",
                    ["Line Chart", "Area Chart", "Sunburst", "Waterfall", "Heatmap Calendar"],
                    horizontal=True, key="adv_chart",
                )

                if adv_chart in ("Line Chart", "Area Chart"):
                    if not all_cols:
                        st.info("No columns available.")
                    else:
                        t1,t2,t3 = st.columns(3)
                        x_time = t1.selectbox("X (time/index)", all_cols, key="adv_x")
                        y_time = t2.multiselect(
                            "Y variables",
                            numeric_cols,
                            default=numeric_cols[:min(2, len(numeric_cols))],
                            key="adv_y",
                        )
                        color_time = t3.selectbox(
                            "Color by (optional)", ["None"]+cat_cols, key="adv_c"
                        )
                        if y_time:
                            plot_df = df[[x_time]+y_time].dropna().sort_values(x_time)
                            if adv_chart == "Line Chart":
                                fig = px.line(
                                    plot_df, x=x_time, y=y_time,
                                    title="Line Chart",
                                    template=plot_template,
                                    markers=True,
                                )
                            else:
                                fig = px.area(
                                    plot_df, x=x_time, y=y_time,
                                    title="Area Chart",
                                    template=plot_template,
                                )
                            st.plotly_chart(fig, use_container_width=True)

                elif adv_chart == "Sunburst":
                    if len(cat_cols) >= 1:
                        sb_path = st.multiselect(
                            "Hierarchy (order matters)",
                            cat_cols,
                            default=cat_cols[:min(2, len(cat_cols))],
                            key="adv_sb_path",
                        )
                        sb_val = st.selectbox(
                            "Value column (optional)", ["Count"]+numeric_cols, key="adv_sb_val"
                        )
                        if sb_path:
                            tmp = df[sb_path].copy()
                            if sb_val == "Count":
                                tmp["_val"] = 1
                                val_col = "_val"
                            else:
                                tmp[sb_val] = df[sb_val]
                                val_col = sb_val
                            tmp = tmp.dropna()
                            fig = px.sunburst(
                                tmp, path=sb_path, values=val_col,
                                title="Sunburst Chart",
                                template=plot_template,
                            )
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("Click segments to zoom in.")
                    else:
                        st.info("Need at least 1 categorical column.")

                elif adv_chart == "Waterfall":
                    if len(numeric_cols) >= 1 and cat_cols:
                        wf1,wf2 = st.columns(2)
                        wf_cat = wf1.selectbox("Category", cat_cols, key="adv_wf_cat")
                        wf_val = wf2.selectbox("Value", numeric_cols, key="adv_wf_val")
                        wf_data = df.groupby(wf_cat)[wf_val].sum().reset_index()
                        wf_data = wf_data.sort_values(wf_val, ascending=False).head(15)
                        fig = go.Figure(go.Waterfall(
                            name="", orientation="v",
                            x=wf_data[wf_cat].astype(str).tolist(),
                            y=wf_data[wf_val].tolist(),
                            connector={"line": {"color": "rgb(63,63,63)"}},
                        ))
                        fig.update_layout(
                            title="Waterfall: " + wf_val + " by " + wf_cat,
                            template=plot_template,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Need at least 1 numeric and 1 categorical column.")

                elif adv_chart == "Heatmap Calendar":
                    st.info(
                        "Select a date column and a numeric value to show "
                        "a calendar-style heatmap."
                    )
                    if all_cols:
                        hc1,hc2 = st.columns(2)
                        date_col = hc1.selectbox("Date column", all_cols, key="adv_hc_date")
                        val_col2 = hc2.selectbox("Value", numeric_cols, key="adv_hc_val")
                        try:
                            tmp = df[[date_col, val_col2]].copy()
                            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
                            tmp = tmp.dropna()
                            tmp["Month"] = tmp[date_col].dt.month_name()
                            tmp["Day"]   = tmp[date_col].dt.day
                            pivot = tmp.groupby(["Month","Day"])[val_col2].mean().unstack()
                            fig = px.imshow(
                                pivot,
                                color_continuous_scale=color_scale,
                                title="Calendar Heatmap: " + val_col2,
                                aspect="auto",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning("Could not create calendar heatmap: " + str(e))


        # ════════════════════════════════════════════════════
        elif active_tab == TAB_OPTIONS[4]:
            st.markdown("## 🤖 Statistical Models")

            MODEL_OPTIONS = [
                "📊 Base Models  (Linear / Logistic / Poisson / Negative Binomial)",
                "🫀 Survival Analysis  (Kaplan-Meier + Cox PH)",
                "📅 Time Series + Interrupted Time Series",
                "🔀 Mixed Effects Models  (LME + GLME)",
                "⚖️ Causal Inference  (PSM + IPW)",
            ]

            selected_model = st.radio(
                "Select model",
                MODEL_OPTIONS,
                key="selected_model",
            )

            st.markdown("---")

            if selected_model == MODEL_OPTIONS[0]:
                render_base_model_tab(df, df_cleaned, plot_template)
            elif selected_model == MODEL_OPTIONS[1]:
                render_survival_tab(df, df_cleaned, plot_template)
            elif selected_model == MODEL_OPTIONS[2]:
                render_time_series_tab(df, df_cleaned, plot_template)
            elif selected_model == MODEL_OPTIONS[3]:
                render_mixed_effects_tab(df, df_cleaned, plot_template)
            elif selected_model == MODEL_OPTIONS[4]:
                render_causal_tab(df, df_cleaned, plot_template)

        elif active_tab == TAB_OPTIONS[5]:
            st.markdown("## Correlation Analysis")
            if len(numeric_cols) >= 2:
                corr_method = st.selectbox("Method", ["pearson", "spearman", "kendall"])
                corr = df[numeric_cols].corr(method=corr_method)

                fig_corr = px.imshow(
                    corr, text_auto=True, aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title=f"Correlation Matrix ({corr_method.capitalize()})",
                )
                fig_corr.update_layout(height=750, template=plot_template)
                st.plotly_chart(fig_corr, use_container_width=True)

                st.markdown("### Correlation Table")
                st.dataframe(corr, use_container_width=True)

                st.markdown("### Strong Correlations")
                pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                pairs = pairs.stack().reset_index()
                pairs.columns = ["Variable 1", "Variable 2", "Correlation"]
                pairs["Abs"] = pairs["Correlation"].abs()
                pairs = pairs.sort_values("Abs", ascending=False)
                strong = pairs[pairs["Abs"] >= strong_corr_threshold]
                if not strong.empty:
                    st.dataframe(strong, use_container_width=True)
                    fig_s = px.bar(
                        strong, x="Abs", y="Variable 1", color="Correlation",
                        orientation="h", title="Strong Correlations",
                        template=plot_template, color_continuous_scale="RdBu_r",
                    )
                    st.plotly_chart(fig_s, use_container_width=True)
                else:
                    st.info(f"No correlations above {strong_corr_threshold}.")
            else:
                st.info("Need at least 2 numeric columns for correlation analysis.")

        # ════════════════════════════════════════════════════
        # Tab 7 — Reports
        # ════════════════════════════════════════════════════
        elif active_tab == TAB_OPTIONS[6]:
            st.markdown("## Download Reports")

            html = generate_html_report(
                uploaded_file_name=uploaded_file.name,
                selected_sheet=selected_sheet,
                total_rows=total_rows,
                total_columns=total_columns,
                total_missing_cells=total_missing_cells,
                missing_cells_percentage=missing_cells_pct,
                duplicate_rows=duplicate_rows,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                possible_id_cols=possible_id_cols,
                high_cardinality_cols=high_cardinality_cols,
                constant_cols=constant_cols,
                date_like_cols=date_like_cols,
                outlier_columns_count=outlier_cols_count,
                missing_report=missing_report,
                dtype_report=dtype_report,
                unique_report=unique_report,
                outlier_report=outlier_report,
                smart_recommendations=smart_recommendations,
            )

            st.markdown("### Full HTML Report")
            st.download_button(
                "📥 Download Full EDA Report (HTML)",
                data=html, file_name="eda_report.html",
                mime="text/html", use_container_width=True,
            )

            st.markdown("---")

            # Excel Export
            st.markdown("### Export All Reports as Excel")
            if st.button("Generate Excel Report", key="excel_export", use_container_width=True):
                try:
                    from io import BytesIO
                    import openpyxl
                    excel_buf = BytesIO()
                    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                        missing_report.to_excel(writer, sheet_name="Missing Values", index=False)
                        dtype_report.to_excel(writer, sheet_name="Column Types", index=False)
                        unique_report.to_excel(writer, sheet_name="Unique Values", index=False)
                        smart_recommendations.to_excel(writer, sheet_name="Recommendations", index=False)
                        if not outlier_report.empty:
                            outlier_report.to_excel(writer, sheet_name="Outliers", index=False)
                        summary_df2 = pd.DataFrame({
                            "Metric": ["Rows","Columns","Missing Cells","Missing %",
                                       "Duplicate Rows","Numeric Columns","Categorical Columns"],
                            "Value":  [total_rows, total_columns, total_missing_cells,
                                       missing_cells_pct, duplicate_rows,
                                       len(numeric_cols), len(categorical_cols)],
                        })
                        summary_df2.to_excel(writer, sheet_name="Summary", index=False)
                    excel_buf.seek(0)
                    st.download_button(
                        "Download Excel Report",
                        data=excel_buf,
                        file_name="eda_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error("Excel export failed: " + str(e))

            st.markdown("---")
            st.markdown("### Smart Recommendations")
            make_download_csv(smart_recommendations, "recommendations.csv",
                              "📥 Download Recommendations")

            st.markdown("---")
            rr1, rr2, rr3 = st.columns(3)
            with rr1:
                st.markdown("### Column Types")
                make_download_csv(dtype_report, "column_types.csv", "📥 Download")
            with rr2:
                st.markdown("### Missing Values")
                make_download_csv(missing_report, "missing_values.csv", "📥 Download")
            with rr3:
                st.markdown("### Unique Values")
                make_download_csv(unique_report, "unique_values.csv", "📥 Download")

            rr4, rr5, rr6 = st.columns(3)
            with rr4:
                st.markdown("### Outliers")
                if not outlier_report.empty:
                    make_download_csv(outlier_report, "outlier_report.csv", "📥 Download")
                else:
                    st.info("No outlier report.")
            with rr5:
                st.markdown("### Original Data")
                st.download_button(
                    "📥 Download original data (CSV)",
                    data=df.to_csv(index=False).encode(),
                    file_name="original_data.csv", mime="text/csv",
                    use_container_width=True,
                )
            with rr6:
                st.markdown("### Summary")
                summary_df = pd.DataFrame({
                    "Metric": [
                        "File Name", "Excel Sheet", "Rows", "Columns",
                        "Missing Cells", "Missing %", "Duplicate Rows",
                        "Numeric Columns", "Categorical Columns",
                        "Possible ID Columns", "High-Cardinality Columns",
                        "Constant Columns", "Date-like Columns", "Outlier Columns",
                    ],
                    "Value": [
                        uploaded_file.name,
                        selected_sheet if selected_sheet else "N/A",
                        total_rows, total_columns, total_missing_cells,
                        missing_cells_pct, duplicate_rows,
                        len(numeric_cols), len(categorical_cols),
                        len(possible_id_cols), len(high_cardinality_cols),
                        len(constant_cols), len(date_like_cols), outlier_cols_count,
                    ],
                })
                summary_df["Value"] = summary_df["Value"].astype(str)
                make_download_csv(summary_df, "summary.csv", "📥 Download")

            st.markdown("### Summary Preview")
            st.dataframe(summary_df, use_container_width=True)

        # ════════════════════════════════════════════════════
        # Tab 8 — Raw Data
        # ════════════════════════════════════════════════════
        elif active_tab == TAB_OPTIONS[7]:
            st.markdown("## Raw Data")
            rows_show = st.slider(
                "Rows to display", 5, min(1000, total_rows), min(50, total_rows)
            )
            st.dataframe(df.head(rows_show), use_container_width=True)

    except Exception as e:
        st.error("An error occurred while processing the file.")
        st.exception(e)

else:
    st.info("Upload a CSV or Excel file from the sidebar to start your analysis.")
    st.markdown("## What you get after uploading")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(
            """<div class="section-card">
            <h3>📊 Dataset Overview</h3>
            </div>""",
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            """<div class="section-card">
            <h3>🧹 Data Cleaning</h3>
            </div>""",
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            """<div class="section-card">
            <h3>🤖 Statistical Modeling</h3>
            </div>""",
            unsafe_allow_html=True,
        )

# Footer
st.markdown(
    """
    <div style="position:fixed;bottom:0;left:0;right:0;background:#0F172A;
        color:#94A3B8;text-align:center;padding:6px 0;font-size:12px;z-index:9999;">
        Developed by <strong style="color:#38BDF8;">Mohammed Asiri</strong>
        &nbsp;&bull;&nbsp; Smart EDA Dashboard
        &nbsp;&bull;&nbsp; All rights reserved &copy; 2026
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Footer — always visible
# ============================================================
st.markdown(
    """
    <style>
    .footer {
        position: fixed; bottom: 0; left: 0; right: 0;
        background: #0F172A; color: #94A3B8;
        text-align: center; padding: 7px 0;
        font-size: 12px; z-index: 9999;
        border-top: 1px solid #1E293B;
    }
    .footer strong { color: #38BDF8; }
    .main { padding-bottom: 40px; }
    </style>
    <div class="footer">
        Developed by <strong>Mohammed Asiri</strong>
        &nbsp;&bull;&nbsp;
        Smart EDA Dashboard &mdash; Statistical Analysis Platform
        &nbsp;&bull;&nbsp;
        All rights reserved &copy; 2026
    </div>
    """,
    unsafe_allow_html=True,
)
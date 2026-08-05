# ============================================================
# Smart EDA Dashboard — app.py
# Main entry point — UI routing only
# All logic lives in modules/
# ============================================================

import streamlit as st

# ── Module imports ───────────────────────────────────────────
from modules.large_data_engine import LARGE_DATA_ROW_THRESHOLD, purge_stale_datasets
from services.error_service import show_friendly_error
from services.quality_service import run_quality_analysis
from services.session_service import initialize_session_state
from services.dataset_service import build_dataset_context
from services.upload_controller import inspect_uploaded_file, load_uploaded_dataset
from ui.visual_page import render_visual_analysis
from ui.model_page import render_model_page
from ui.cleaning_page import render_cleaning_page
from ui.quality_page import render_quality_page
from ui.reports_page import render_reports_page
from ui.raw_data_page import render_raw_data_page
from ui.overview_page import render_overview_page
from ui.correlation_page import render_correlation_page
from ui.sidebar import render_dataset_sidebar
from ui.layout import (
    TAB_OPTIONS,
    render_first_file_guide,
    render_global_controls,
    render_hero,
    render_navigation,
)


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

    /* Main navigation (st.radio styled as horizontal tabs) */
    div[role="radiogroup"] {
        gap: 8px;
        flex-wrap: wrap;
    }
    div[role="radiogroup"] label {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 10px 16px;
        border: 1px solid #E2E8F0;
        margin-right: 0 !important;
        transition: all 0.15s ease;
    }
    div[role="radiogroup"] label:hover {
        border-color: #2563EB;
        background-color: #F0F7FF;
    }
    /* Hide the circular radio indicator, keep only the text label */
    div[role="radiogroup"] label > div[data-baseweb="radio"] > div:first-of-type {
        width: 0;
        height: 0;
        margin: 0;
        opacity: 0;
        position: absolute;
    }
    div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        font-weight: 600;
        font-size: 14px;
        color: #0F172A;
        margin: 0;
    }
    div[role="radiogroup"] label[aria-checked="true"] {
        background-color: var(--app-primary-color, #2563EB);
        border-color: var(--app-primary-color, #2563EB);
    }
    div[role="radiogroup"] label[aria-checked="true"] div[data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
    }

    div[data-testid="stFileUploader"] label  { color: #0F172A !important; font-weight: 700 !important; font-size: 16px !important; }
    div[data-testid="stFileUploader"] section { background-color: #F8FAFC !important; border: 2px dashed #2563EB !important; border-radius: 14px !important; }
    div[data-testid="stFileUploader"] button  { color: #FFFFFF !important; background-color: #2563EB !important; border: 1px solid #2563EB !important; font-weight: 700 !important; }
    div[data-testid="stFileUploader"] button:hover { background-color: #1D4ED8 !important; }
    div[data-testid="stFileUploader"] small { color: #334155 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


initialize_session_state(st.session_state)
if not st.session_state["temp_cleanup_done"]:
    purge_stale_datasets()
    st.session_state["temp_cleanup_done"] = True


# Quality execution is provided by services.quality_service.

def require_quality_analysis(
    df, numeric_cols, categorical_cols, cardinality_threshold
):
    """Render an explicit run control and stop the page until results exist."""
    if st.session_state["quality_analysis"] is not None:
        return st.session_state["quality_analysis"]

    st.info(
        "Quality analysis has not run yet. It uses the complete dataset and is "
        "started explicitly so navigation does not repeat expensive work."
    )
    if st.button("Run full quality analysis", type="primary", key="run_quality"):
        with st.spinner("Analyzing all rows..."):
            st.session_state["quality_analysis"] = run_quality_analysis(
                df,
                numeric_cols,
                categorical_cols,
                cardinality_threshold,
            )
        st.rerun()
    st.stop()


# Visualization helpers are provided by services.visual_service.

render_hero()
layout_controls = render_global_controls()
uploaded_file = layout_controls.uploaded_file
plot_template = layout_controls.plot_template
primary_color = layout_controls.primary_color
color_scale = layout_controls.color_scale
high_cardinality_threshold = layout_controls.high_cardinality_threshold
strong_corr_threshold = layout_controls.strong_corr_threshold


# ============================================================
# Main app
# ============================================================
if uploaded_file is not None:
    try:
        upload_candidate = inspect_uploaded_file(uploaded_file)
    except ValueError as validation_error:
        st.error(str(validation_error))
        st.stop()

    try:
        selected_sheet = None
        if upload_candidate.sheet_names:
            selected_sheet = st.sidebar.selectbox(
                "Choose Excel sheet", upload_candidate.sheet_names
            )
        loaded_dataset = load_uploaded_dataset(
            upload_candidate, selected_sheet, st.session_state
        )
        file_id = loaded_dataset.file_id
        df = loaded_dataset.dataframe
        dataset_store = loaded_dataset.dataset_store

        context = build_dataset_context(
            df, dataset_store, st.session_state, high_cardinality_threshold
        )
        numeric_cols = context.numeric_cols
        categorical_cols = context.categorical_cols
        total_rows = context.total_rows
        total_columns = context.total_columns
        total_missing_cells = context.total_missing_cells
        missing_cells_pct = context.missing_cells_pct
        duplicate_rows = context.duplicate_rows
        dtype_report = context.dtype_report
        quality_analysis = context.quality_analysis
        possible_id_cols = context.possible_id_cols
        high_cardinality_cols = context.high_cardinality_cols
        constant_cols = context.constant_cols
        date_like_cols = context.date_like_cols
        outlier_report = context.outlier_report
        missing_report = context.missing_report
        unique_report = context.unique_report
        smart_recommendations = context.smart_recommendations
        outlier_cols_count = context.outlier_cols_count

        if render_dataset_sidebar(
            uploaded_file=uploaded_file,
            selected_sheet=selected_sheet,
            total_rows=total_rows,
            total_columns=total_columns,
            numeric_count=len(numeric_cols),
            categorical_count=len(categorical_cols),
            is_large=dataset_store.is_large,
            large_data_threshold=LARGE_DATA_ROW_THRESHOLD,
        ):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.success("✅ File loaded. EDA report is ready.")

        st.markdown("## Dataset Snapshot")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{total_rows:,}")
        c2.metric("Columns", f"{total_columns:,}")
        c3.metric("Missing Cells", f"{total_missing_cells:,}", f"{missing_cells_pct}%")
        c4.metric(
            "Duplicate Rows",
            f"{duplicate_rows:,}" if duplicate_rows is not None else "Not run",
        )

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Numeric Columns", len(numeric_cols))
        c6.metric("Categorical Columns", len(categorical_cols))
        c7.metric(
            "Possible ID Columns",
            len(possible_id_cols) if quality_analysis is not None else "Not run",
        )
        c8.metric(
            "Outlier Columns",
            outlier_cols_count if quality_analysis is not None else "Not run",
        )

        st.markdown("---")

        if st.session_state["df_cleaned"] is None:
            st.session_state["df_cleaned"] = df.copy()
        df_cleaned = st.session_state["df_cleaned"]

        render_first_file_guide(file_id)
        active_tab = render_navigation()

        if active_tab in {TAB_OPTIONS[1], TAB_OPTIONS[2], TAB_OPTIONS[6]}:
            quality_analysis = require_quality_analysis(
                df,
                numeric_cols,
                categorical_cols,
                high_cardinality_threshold,
            )
            duplicate_rows = quality_analysis["duplicate_rows"]
            possible_id_cols = quality_analysis["possible_id_cols"]
            high_cardinality_cols = quality_analysis["high_cardinality_cols"]
            constant_cols = quality_analysis["constant_cols"]
            date_like_cols = quality_analysis["date_like_cols"]
            outlier_report = quality_analysis["outlier_report"]
            missing_report = quality_analysis["missing_report"]
            dtype_report = quality_analysis["dtype_report"]
            unique_report = quality_analysis["unique_report"]
            smart_recommendations = quality_analysis["smart_recommendations"]
            outlier_cols_count = int(
                (outlier_report["Outliers Count"] > 0).sum()
            ) if not outlier_report.empty else 0

        st.markdown("---")

        if active_tab == TAB_OPTIONS[0]:
            render_overview_page(df, dtype_report, plot_template)

        # ════════════════════════════════════════════════════
        # Tab 2 — Data Quality
        # ════════════════════════════════════════════════════
        elif active_tab == TAB_OPTIONS[1]:
            render_quality_page(
                df=df,
                quality_analysis=quality_analysis,
                numeric_cols=numeric_cols,
                missing_report=missing_report,
                missing_cells_pct=missing_cells_pct,
                duplicate_rows=duplicate_rows,
                possible_id_cols=possible_id_cols,
                high_cardinality_cols=high_cardinality_cols,
                constant_cols=constant_cols,
                date_like_cols=date_like_cols,
                outlier_report=outlier_report,
                unique_report=unique_report,
                smart_recommendations=smart_recommendations,
                plot_template=plot_template,
            )

        elif active_tab == TAB_OPTIONS[2]:
            render_cleaning_page(
                df=df,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                constant_cols=constant_cols,
                possible_id_cols=possible_id_cols,
                total_missing_cells=total_missing_cells,
                duplicate_rows=duplicate_rows,
            )

        elif active_tab == TAB_OPTIONS[3]:
            render_visual_analysis(
                df, dataset_store, plot_template, primary_color, color_scale
            )

        elif active_tab == TAB_OPTIONS[4]:
            render_model_page(df, df_cleaned, plot_template)

        elif active_tab == TAB_OPTIONS[5]:
            render_correlation_page(
                df, numeric_cols, strong_corr_threshold, plot_template
            )

        # ════════════════════════════════════════════════════
        # Tab 7 — Reports
        # ════════════════════════════════════════════════════
        elif active_tab == TAB_OPTIONS[6]:
            render_reports_page(
                df=df,
                uploaded_file=uploaded_file,
                selected_sheet=selected_sheet,
                total_rows=total_rows,
                total_columns=total_columns,
                total_missing_cells=total_missing_cells,
                missing_cells_pct=missing_cells_pct,
                duplicate_rows=duplicate_rows,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                possible_id_cols=possible_id_cols,
                high_cardinality_cols=high_cardinality_cols,
                constant_cols=constant_cols,
                date_like_cols=date_like_cols,
                outlier_cols_count=outlier_cols_count,
                missing_report=missing_report,
                dtype_report=dtype_report,
                unique_report=unique_report,
                outlier_report=outlier_report,
                smart_recommendations=smart_recommendations,
            )

        elif active_tab == TAB_OPTIONS[7]:
            render_raw_data_page(df)

    except Exception as e:
        show_friendly_error(e, context="processing this file")

else:
    st.info("Upload a file to start your analysis.")
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

"""Shared dashboard layout, upload controls, theme, guidance, and navigation."""

from dataclasses import dataclass

import streamlit as st

from services.file_service import SUPPORTED_EXTENSIONS


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

THEME_COLORS = {
    "Blue": {"primary": "#2563EB", "scale": "Blues", "template": "plotly_white"},
    "Green": {"primary": "#16A34A", "scale": "Greens", "template": "plotly_white"},
    "Purple": {"primary": "#7C3AED", "scale": "Purples", "template": "plotly_white"},
    "Orange": {"primary": "#EA580C", "scale": "Oranges", "template": "plotly_white"},
    "Red": {"primary": "#DC2626", "scale": "Reds", "template": "plotly_white"},
}


@dataclass(frozen=True)
class LayoutControls:
    uploaded_file: object
    plot_template: str
    primary_color: str
    color_scale: str
    high_cardinality_threshold: int
    strong_corr_threshold: float


def render_hero():
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


def render_global_controls():
    """Render upload and global analysis controls and return their values."""
    st.sidebar.markdown("## 📊 Smart EDA")
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=SUPPORTED_EXTENSIONS,
        key="main_uploader",
        label_visibility="collapsed",
    )
    st.markdown(
        """
        <style>
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderDropzoneInstructions"] small { display: none !important; }
        [data-testid="stFileUploaderDropzoneInstructions"]::after {
            content: "All formats supported"; font-size: 14px; color: #64748B;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    theme_choice = st.sidebar.selectbox(
        "Color Theme", list(THEME_COLORS), key="color_theme"
    )
    theme = THEME_COLORS[theme_choice]
    primary = theme["primary"]
    red, green, blue = (int(primary[i:i + 2], 16) for i in (1, 3, 5))
    tint = f"rgba({red},{green},{blue},0.08)"
    border = f"rgba({red},{green},{blue},0.35)"
    st.markdown(
        f"""
        <style>
        div[role="radiogroup"] label {{ --app-primary-color: {primary}; }}
        div[role="radiogroup"] label:hover {{
            border-color: {primary} !important; background-color: {tint} !important;
        }}
        div[role="radiogroup"] label[aria-checked="true"] {{
            background-color: {primary} !important; border-color: {primary} !important;
        }}
        div[data-testid="stFileUploader"] button {{
            background-color: {primary} !important; border-color: {primary} !important;
        }}
        .hero-container {{
            background: linear-gradient(135deg, #0F172A 0%, {primary} 100%) !important;
        }}
        .guide-box {{ background: {tint} !important; border: 1px solid {border} !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    high_cardinality = st.sidebar.slider(
        "High-cardinality threshold", 10, 200, 50, 10
    )
    strong_correlation = st.sidebar.slider(
        "Strong correlation threshold", 0.30, 0.90, 0.50, 0.05
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### What this app does")
    st.sidebar.markdown(
        """
- Dataset overview and deep profiling
- Data quality checks and cleaning
- Statistical tests and models
- Interactive visualizations and correlations
- Full HTML, CSV, and Excel reports
        """
    )
    return LayoutControls(
        uploaded_file, theme["template"], primary, theme["scale"],
        high_cardinality, strong_correlation,
    )


def render_first_file_guide(file_id):
    """Show the suggested workflow once for each uploaded dataset."""
    if st.session_state.get("guide_dismissed_for") == file_id:
        return
    st.markdown(
        """
        <div class="guide-box" style="border-radius:12px;padding:16px 20px;margin-bottom:10px;">
        <strong>New here? Suggested path:</strong><br>
        1. <strong>Overview</strong> &mdash; inspect the data<br>
        2. <strong>Data Quality</strong> &mdash; identify issues<br>
        3. <strong>Data Cleaning</strong> &mdash; optionally fix selected issues<br>
        4. <strong>Models</strong> &mdash; run the appropriate full-data analysis
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Got it, hide this", key="dismiss_guide"):
        st.session_state["guide_dismissed_for"] = file_id
        st.rerun()


def render_navigation():
    return st.radio(
        "Navigation",
        TAB_OPTIONS,
        horizontal=True,
        key="active_tab",
        label_visibility="collapsed",
    )

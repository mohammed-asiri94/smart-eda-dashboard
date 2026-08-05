"""Statistical-model page routing and explicit missing-row policy."""

import streamlit as st

from modules.models.base_models import render_base_model_tab
from modules.models.causal import render_causal_tab
from modules.models.machine_learning import render_machine_learning_tab
from modules.models.mixed_effects import render_mixed_effects_tab
from modules.models.survival import render_survival_tab
from modules.models.time_series import render_time_series_tab


MODEL_OPTIONS = [
    "📊 Base Models  (Linear / Logistic / Poisson / Negative Binomial)",
    "🫀 Survival Analysis  (Kaplan-Meier + Cox PH)",
    "📅 Time Series + Interrupted Time Series",
    "🔀 Mixed Effects Models  (LME + GLME)",
    "⚖️ Causal Inference  (PSM + IPW)",
    "🤖 Machine Learning  (Classification / Regression / Clustering)",
]


def render_model_page(df, df_cleaned, plot_template):
    """Render model selection while preserving full-data safeguards."""
    st.markdown("## 🤖 Statistical Models")
    st.info(
        f"Full-data modeling is enabled: all {len(df):,} rows are passed "
        "to the selected model. No sampling is applied, and no row is "
        "temporarily excluded unless you explicitly allow it below."
    )
    missing_policy_label = st.radio(
        "If required model values are missing or invalid",
        [
            "Stop and ask me to clean the data",
            "Temporarily exclude ineligible rows for this model",
        ],
        horizontal=True,
        key="model_missing_policy_label",
        help=(
            "Temporary exclusion affects only the current model input. It never "
            "deletes rows from the original or cleaned dataset."
        ),
    )
    st.session_state["model_missing_policy"] = (
        "allow_temporary_exclusion"
        if missing_policy_label.startswith("Temporarily")
        else "stop"
    )
    if st.session_state["model_missing_policy"] == "stop":
        st.caption(
            "Safe default: the model will stop if even one row would need to be excluded."
        )
    else:
        st.warning(
            "You allowed temporary model-only exclusion. The audit will show "
            "the exact number of affected rows before results."
        )

    if st.session_state.get("selected_model_radio_v2") not in MODEL_OPTIONS:
        st.session_state["selected_model_radio_v2"] = MODEL_OPTIONS[0]
    selected_model = st.radio(
        "Select model", MODEL_OPTIONS, key="selected_model_radio_v2"
    )
    st.markdown("---")

    renderers = [
        render_base_model_tab,
        render_survival_tab,
        render_time_series_tab,
        render_mixed_effects_tab,
        render_causal_tab,
        render_machine_learning_tab,
    ]
    renderers[MODEL_OPTIONS.index(selected_model)](df, df_cleaned, plot_template)

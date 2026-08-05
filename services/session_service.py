"""Centralized Streamlit session-state lifecycle."""


DEFAULT_SESSION_STATE = {
    "df_cleaned": None,
    "cleaning_applied": False,
    "last_file_id": None,
    "dataset_store": None,
    "dataset_core_metrics": None,
    "original_dataframe": None,
    "quality_analysis": None,
    "overview_summary": None,
    "profiling_enabled": False,
    "zscore_analysis": None,
    "visual_sample": None,
    "visual_sample_key": None,
    "temp_cleanup_done": False,
    "selected_model_radio_v2": "📊 Base Models  (Linear / Logistic / Poisson / Negative Binomial)",
}

DATASET_SCOPED_KEYS = (
    "df_cleaned",
    "cleaning_applied",
    "dataset_store",
    "dataset_core_metrics",
    "original_dataframe",
    "quality_analysis",
    "overview_summary",
    "profiling_enabled",
    "zscore_analysis",
    "visual_sample",
    "visual_sample_key",
)


def initialize_session_state(state):
    """Add missing defaults without overwriting existing user state."""
    for key, value in DEFAULT_SESSION_STATE.items():
        state.setdefault(key, value)


def reset_for_dataset(state, file_id):
    """Invalidate only results tied to the previous uploaded dataset."""
    if state.get("last_file_id") == file_id:
        return False
    for key in DATASET_SCOPED_KEYS:
        state[key] = DEFAULT_SESSION_STATE[key]
    state["last_file_id"] = file_id
    return True

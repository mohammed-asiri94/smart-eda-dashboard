"""Safe user-facing errors with rotating server-side diagnostics."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import tempfile
import uuid

import streamlit as st


LOG_PATH = Path(tempfile.gettempdir()) / "smart_eda_dashboard.log"
logger = logging.getLogger("smart_eda")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        LOG_PATH, maxBytes=1_000_000, backupCount=2, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(handler)


ERROR_TRANSLATIONS = [
    ("perfect separation", "Your treatment/outcome variable may be too predictable from the covariates. Try removing one strongly correlated covariate."),
    ("singular matrix", "Some selected columns are perfectly correlated. Try removing one of them."),
    ("convergence", "The model could not converge. Try a simpler model or fewer covariates."),
    ("could not convert string", "A selected column contains text where a number was expected."),
    ("not enough values to unpack", "There is not enough valid data after applying the selected requirements."),
    ("division by zero", "A required column may have no variation."),
    ("no module named", "A required library is missing. Install requirements.txt."),
    ("memory", "The operation exceeded available memory."),
    ("index out of range", "The selected columns do not have enough valid data."),
    ("must be exactly", "This model requires a specific number of categories."),
]


def friendly_error_message(error_text):
    lower = str(error_text).lower()
    for keyword, friendly in ERROR_TRANSLATIONS:
        if keyword in lower:
            return friendly
    return None


def show_friendly_error(exception, context=""):
    error_id = uuid.uuid4().hex[:8]
    friendly = friendly_error_message(exception)
    logger.exception(
        "Error ID %s while %s", error_id, context or "processing a dashboard step"
    )
    st.error(
        "Something went wrong while " + context + "."
        if context
        else "Something went wrong while processing this step."
    )
    if friendly:
        st.info("Likely cause: " + friendly)
    st.caption(f"Error reference: {error_id}")

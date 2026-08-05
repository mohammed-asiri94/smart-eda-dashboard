"""Raw-data preview page."""

import streamlit as st


def render_raw_data_page(df):
    """Display a bounded browser preview without changing model inputs."""
    st.markdown("## Raw Data")
    total_rows = len(df)
    rows_show = st.slider(
        "Rows to display", 5, min(1000, total_rows), min(50, total_rows)
    )
    st.dataframe(df.head(rows_show), use_container_width=True)

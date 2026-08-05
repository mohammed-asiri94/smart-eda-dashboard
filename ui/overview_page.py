"""Dataset overview and opt-in profiling page."""

import plotly.express as px
import streamlit as st

from modules.data_profiling import render_data_profiling


def render_overview_page(df, dtype_report, plot_template):
    """Render a lightweight overview; expensive summaries remain user-triggered."""
    st.markdown("## Overview")
    st.markdown("### Data Preview (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Column Information")
    st.dataframe(dtype_report, use_container_width=True)

    st.markdown("### Summary Statistics")
    if st.session_state["overview_summary"] is None:
        if st.button("Generate full summary statistics", key="run_summary"):
            with st.spinner("Calculating summary statistics on all rows..."):
                st.session_state["overview_summary"] = df.describe(include="all").T
            st.rerun()
        else:
            st.info("Summary statistics run only when requested.")
    else:
        st.dataframe(st.session_state["overview_summary"], use_container_width=True)

    st.markdown("---")
    st.markdown("### Data Profiling")
    if not st.session_state["profiling_enabled"]:
        if st.button("Start full data profiling", key="run_profiling"):
            st.session_state["profiling_enabled"] = True
            st.rerun()
        else:
            st.info("Deep profiling runs only when requested.")
    else:
        render_data_profiling(df, plot_template)

    st.markdown("### Column Type Distribution")
    type_counts = dtype_report["Data Type"].value_counts().reset_index()
    type_counts.columns = ["Data Type", "Count"]
    fig = px.bar(
        type_counts,
        x="Data Type",
        y="Count",
        text="Count",
        title="Column Data Types",
        template=plot_template,
    )
    st.plotly_chart(fig, use_container_width=True)

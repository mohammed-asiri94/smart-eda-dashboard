"""Data-quality analysis page."""

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_quality import calculate_outliers_zscore, get_data_quality_status
from modules.statistical_tests import render_statistical_tests


def render_quality_page(
    df,
    quality_analysis,
    numeric_cols,
    missing_report,
    missing_cells_pct,
    duplicate_rows,
    possible_id_cols,
    high_cardinality_cols,
    constant_cols,
    date_like_cols,
    outlier_report,
    unique_report,
    smart_recommendations,
    plot_template,
):
    """Render cached/on-demand quality results without modifying the dataset."""
    st.markdown("## Data Quality Report")
    st.caption(
        f"Analyzed all {quality_analysis['rows_analyzed']:,} rows in "
        f"{quality_analysis['elapsed_seconds']:.2f} seconds. Results are cached "
        "for this dataset and configuration."
    )

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
        st.dataframe(
            quality_analysis["duplicate_preview"], use_container_width=True
        )
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
            zscore_state = st.session_state["zscore_analysis"]
            if st.button("Run Z-Score analysis", key="run_zscore"):
                with st.spinner("Calculating Z-Scores on all rows..."):
                    active_outlier = calculate_outliers_zscore(
                        df, tuple(numeric_cols), threshold=z_thresh
                    )
                    st.session_state["zscore_analysis"] = {
                        "threshold": z_thresh,
                        "report": active_outlier,
                    }
                st.rerun()
            elif zscore_state is not None:
                active_outlier = zscore_state["report"]
                if zscore_state["threshold"] != z_thresh:
                    st.warning(
                        "The displayed result uses threshold "
                        f"{zscore_state['threshold']}. Run again to apply {z_thresh}."
                    )
            else:
                active_outlier = pd.DataFrame()
                st.info("Choose a threshold, then run the Z-Score analysis.")
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


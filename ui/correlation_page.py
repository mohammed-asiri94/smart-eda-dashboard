"""Full-data correlation analysis page."""

import numpy as np
import plotly.express as px
import streamlit as st


def render_correlation_page(df, numeric_cols, strong_corr_threshold, plot_template):
    """Calculate correlations from all valid pairs; no sampling is applied."""
    st.markdown("## Correlation Analysis")
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns for correlation analysis.")
        return

    corr_method = st.selectbox("Method", ["pearson", "spearman", "kendall"])
    corr = df[numeric_cols].corr(method=corr_method)
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
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
    if strong.empty:
        st.info(f"No correlations above {strong_corr_threshold}.")
        return

    st.dataframe(strong, use_container_width=True)
    fig_s = px.bar(
        strong,
        x="Abs",
        y="Variable 1",
        color="Correlation",
        orientation="h",
        title="Strong Correlations",
        template=plot_template,
        color_continuous_scale="RdBu_r",
    )
    st.plotly_chart(fig_s, use_container_width=True)

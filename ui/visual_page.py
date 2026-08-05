"""Visual Analysis page isolated from the main Streamlit router."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.visual_service import (
    MAX_MULTIVARIATE_POINTS,
    aggregate_line_display,
    exact_density_figure,
    exact_histogram_figure,
    get_visual_sample,
    show_visual_sampling_notice,
)


def render_visual_analysis(df, dataset_store, plot_template, primary_color, color_scale):
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
                fig = exact_histogram_figure(
                    df[sel], sel, plot_template, primary_color, bins=40
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Histogram bins are calculated from all valid rows.")

            elif chart_type == "Boxplot":
                grp_box = st.selectbox(
                    "Group by (optional)", ["None"] + cat_cols, key="uni_box_grp"
                )
                chart_df = get_visual_sample(
                    df,
                    dataset_store.dataset_id,
                    stratify_col=None if grp_box == "None" else grp_box,
                )
                fig = px.box(
                    chart_df,
                    y=sel,
                    x=None if grp_box == "None" else grp_box,
                    color=None if grp_box == "None" else grp_box,
                    points="outliers",
                    title="Boxplot of " + sel,
                    template=plot_template,
                )
                st.plotly_chart(fig, use_container_width=True)
                show_visual_sampling_notice(len(df), len(chart_df))

            elif chart_type == "Violin":
                grp_vio = st.selectbox(
                    "Group by (optional)", ["None"] + cat_cols, key="uni_vio_grp"
                )
                chart_df = get_visual_sample(
                    df,
                    dataset_store.dataset_id,
                    stratify_col=None if grp_vio == "None" else grp_vio,
                )
                fig = px.violin(
                    chart_df,
                    y=sel,
                    x=None if grp_vio == "None" else grp_vio,
                    color=None if grp_vio == "None" else grp_vio,
                    box=True, points="outliers",
                    title="Violin Plot of " + sel,
                    template=plot_template,
                )
                st.plotly_chart(fig, use_container_width=True)
                show_visual_sampling_notice(len(df), len(chart_df))

            elif chart_type == "ECDF":
                valid_values = pd.to_numeric(
                    df[sel], errors="coerce"
                ).dropna().sort_values()
                ecdf_points = min(2_000, len(valid_values))
                probabilities = np.linspace(0, 1, ecdf_points)
                quantiles = valid_values.quantile(probabilities).to_numpy()
                fig = go.Figure(
                    go.Scatter(
                        x=quantiles,
                        y=probabilities,
                        mode="lines",
                        line={"color": primary_color},
                    )
                )
                fig.update_layout(
                    title="ECDF of " + sel + " (all rows)",
                    xaxis_title=sel,
                    yaxis_title="Cumulative probability",
                    template=plot_template,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Quantiles are calculated from all valid rows and compressed "
                    "to at most 2,000 display points."
                )

            elif chart_type == "Strip Plot":
                grp_str = st.selectbox(
                    "Group by (optional)", ["None"] + cat_cols, key="uni_str_grp"
                )
                chart_df = get_visual_sample(
                    df,
                    dataset_store.dataset_id,
                    stratify_col=None if grp_str == "None" else grp_str,
                )
                fig = px.strip(
                    chart_df,
                    y=sel,
                    x=None if grp_str == "None" else grp_str,
                    color=None if grp_str == "None" else grp_str,
                    title="Strip Plot of " + sel,
                    template=plot_template,
                )
                st.plotly_chart(fig, use_container_width=True)
                show_visual_sampling_notice(len(df), len(chart_df))

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
                chart_df = get_visual_sample(
                    df,
                    dataset_store.dataset_id,
                    stratify_col=color_arg,
                )
                fig = px.scatter(
                    chart_df, x=x_col, y=y_col, color=color_arg,
                    trendline="ols",
                    trendline_color_override="red",
                    title=x_col + " vs " + y_col,
                    template=plot_template,
                    opacity=0.6,
                )
                st.plotly_chart(fig, use_container_width=True)
                show_visual_sampling_notice(len(df), len(chart_df))
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
                fig = exact_density_figure(
                    df, x_col, y_col, plot_template, color_scale
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("All valid rows are counted in the 30 × 30 density grid.")

            elif chart_type == "2D Histogram":
                fig = exact_density_figure(
                    df,
                    x_col,
                    y_col,
                    plot_template,
                    color_scale,
                    contour=True,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Contours are calculated from all valid rows.")

            elif chart_type == "Bubble":
                if len(numeric_cols) >= 3:
                    size_col = st.selectbox(
                        "Bubble size", numeric_cols, key="biv_bubble_size"
                    )
                    chart_df = get_visual_sample(
                        df,
                        dataset_store.dataset_id,
                        stratify_col=color_arg,
                    )
                    fig = px.scatter(
                        chart_df, x=x_col, y=y_col,
                        size=size_col, color=color_arg,
                        title="Bubble: " + x_col + " vs " + y_col,
                        template=plot_template,
                        size_max=40, opacity=0.6,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    show_visual_sampling_notice(len(df), len(chart_df))
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
                    chart_df = get_visual_sample(
                        df, dataset_store.dataset_id, stratify_col=cat_col
                    )
                    fig = px.box(
                        chart_df, x=cat_col, y=num_col,
                        color=cat_col, points="outliers",
                        title=num_col + " by " + cat_col,
                        template=plot_template,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    show_visual_sampling_notice(len(df), len(chart_df))

                elif chart_type == "Grouped Violin":
                    chart_df = get_visual_sample(
                        df, dataset_store.dataset_id, stratify_col=cat_col
                    )
                    fig = px.violin(
                        chart_df, x=cat_col, y=num_col,
                        color=cat_col, box=True, points="outliers",
                        title=num_col + " by " + cat_col,
                        template=plot_template,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    show_visual_sampling_notice(len(df), len(chart_df))

                elif chart_type == "Strip Plot":
                    chart_df = get_visual_sample(
                        df, dataset_store.dataset_id, stratify_col=cat_col
                    )
                    fig = px.strip(
                        chart_df, x=cat_col, y=num_col, color=cat_col,
                        title=num_col + " by " + cat_col,
                        template=plot_template,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    show_visual_sampling_notice(len(df), len(chart_df))

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
                    top_cat1 = set(
                        df[cat1].astype(str).value_counts().head(30).index
                    )
                    top_cat2 = set(
                        df[cat2].astype(str).value_counts().head(30).index
                    )
                    bounded_cat1 = df[cat1].astype(str).where(
                        df[cat1].astype(str).isin(top_cat1), "Other"
                    )
                    bounded_cat2 = df[cat2].astype(str).where(
                        df[cat2].astype(str).isin(top_cat2), "Other"
                    )
                    ct = pd.crosstab(bounded_cat1, bounded_cat2)
                    ct.index.name = cat1
                    ct.columns.name = cat2
                    chart_type = st.radio(
                        "Chart",
                        ["Heatmap", "Grouped Bar", "Stacked Bar"],
                        horizontal=True,
                        key="biv_cc_chart",
                    )

                    if chart_type == "Heatmap":
                        fig = px.imshow(
                            ct,
                            text_auto=True,
                            aspect="auto",
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
                    st.caption(
                        "Counts use all rows. Each axis keeps its 30 most common "
                        "categories and combines the remainder as Other."
                    )

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
                    chart_df = get_visual_sample(
                        df,
                        dataset_store.dataset_id,
                        max_points=MAX_MULTIVARIATE_POINTS,
                        stratify_col=None if color_grp == "None" else color_grp,
                    )
                    fig = px.scatter_matrix(
                        chart_df,
                        dimensions=sel_cols[:max_cols],
                        color=None if color_grp == "None" else color_grp,
                        title="Pair Plot",
                        template=plot_template,
                        opacity=0.5,
                    )
                    fig.update_traces(diagonal_visible=True)
                    fig.update_layout(height=700)
                    st.plotly_chart(fig, use_container_width=True)
                    show_visual_sampling_notice(len(df), len(chart_df))
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
                    chart_df = get_visual_sample(
                        df,
                        dataset_store.dataset_id,
                        max_points=MAX_MULTIVARIATE_POINTS,
                    )
                    fig = px.parallel_coordinates(
                        chart_df.dropna(subset=sel_cols),
                        dimensions=sel_cols,
                        color=color_col_pc,
                        color_continuous_scale=color_scale,
                        title="Parallel Coordinates",
                        template=plot_template,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    show_visual_sampling_notice(len(df), len(chart_df))
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
                chart_df = get_visual_sample(
                    df,
                    dataset_store.dataset_id,
                    max_points=MAX_MULTIVARIATE_POINTS,
                )
                fig = px.scatter_matrix(
                    chart_df, dimensions=sel_cols,
                    title="Scatter Matrix",
                    template=plot_template,
                    opacity=0.5,
                )
                fig.update_layout(height=650)
                st.plotly_chart(fig, use_container_width=True)
                show_visual_sampling_notice(len(df), len(chart_df))

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
                    top_radar_groups = (
                        df[radar_cat].value_counts(dropna=False).head(20).index
                    )
                    grp_means = (
                        df[df[radar_cat].isin(top_radar_groups)]
                        .groupby(radar_cat, observed=True)[radar_cols]
                        .mean()
                    )
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
                    st.caption(
                        "Means use all rows in the 20 largest groups; values are "
                        "normalized 0-1 for comparison."
                    )
            else:
                st.info("Need at least 3 numeric columns and 1 categorical column.")

        elif mv_chart == "3D Scatter":
            if len(numeric_cols) >= 3:
                d1,d2,d3 = st.columns(3)
                x3 = d1.selectbox("X", numeric_cols, key="mv_3dx")
                y3 = d2.selectbox("Y", numeric_cols, key="mv_3dy")
                z3 = d3.selectbox("Z", numeric_cols, key="mv_3dz")
                c3 = st.selectbox("Color", ["None"]+cat_cols+numeric_cols, key="mv_3dc")
                chart_df = get_visual_sample(
                    df,
                    dataset_store.dataset_id,
                    max_points=MAX_MULTIVARIATE_POINTS,
                    stratify_col=c3 if c3 in cat_cols else None,
                )
                fig = px.scatter_3d(
                    chart_df.dropna(subset=[x3,y3,z3]),
                    x=x3, y=y3, z=z3,
                    color=None if c3=="None" else c3,
                    title="3D Scatter: " + x3 + " / " + y3 + " / " + z3,
                    template=plot_template,
                    opacity=0.6,
                )
                fig.update_layout(height=650)
                st.plotly_chart(fig, use_container_width=True)
                show_visual_sampling_notice(len(df), len(chart_df))
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
                    color_column = None if color_time == "None" else color_time
                    plot_df = aggregate_line_display(
                        df, x_time, y_time, color_column=color_column
                    )
                    if color_column:
                        display_df = plot_df.melt(
                            id_vars=[x_time, color_column],
                            value_vars=y_time,
                            var_name="Variable",
                            value_name="Value",
                        )
                        chart_args = {
                            "data_frame": display_df,
                            "x": x_time,
                            "y": "Value",
                            "color": color_column,
                            "line_group": "Variable",
                        }
                    else:
                        chart_args = {
                            "data_frame": plot_df,
                            "x": x_time,
                            "y": y_time,
                        }
                    if adv_chart == "Line Chart":
                        fig = px.line(
                            **chart_args,
                            title="Line Chart",
                            template=plot_template,
                            markers=True,
                        )
                    else:
                        fig = px.area(
                            **chart_args,
                            title="Area Chart",
                            template=plot_template,
                        )
                    st.plotly_chart(fig, use_container_width=True)
                    if len(plot_df) < len(df):
                        st.caption(
                            f"All {len(df):,} rows are represented by "
                            f"{len(plot_df):,} ordered mean buckets."
                        )

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
                    if sb_val == "Count":
                        tmp = (
                            df.groupby(sb_path, dropna=False, observed=True)
                            .size()
                            .reset_index(name="_val")
                        )
                        val_col = "_val"
                    else:
                        tmp = (
                            df.groupby(sb_path, dropna=False, observed=True)[sb_val]
                            .sum(min_count=1)
                            .reset_index()
                        )
                        val_col = sb_val
                    tmp = tmp.dropna()
                    combinations = len(tmp)
                    if combinations > 5_000:
                        tmp = tmp.nlargest(5_000, val_col)
                    fig = px.sunburst(
                        tmp, path=sb_path, values=val_col,
                        title="Sunburst Chart",
                        template=plot_template,
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    if combinations > len(tmp):
                        st.caption(
                            f"Aggregated from all rows; displaying the top "
                            f"{len(tmp):,} of {combinations:,} hierarchy combinations."
                        )
                    else:
                        st.caption("Aggregated from all rows. Click segments to zoom in.")
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

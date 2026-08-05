"""Display-only sampling and full-data visual aggregations."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


MAX_VISUAL_POINTS = 20_000
MAX_MULTIVARIATE_POINTS = 5_000


def create_visual_sample(df, max_points=MAX_VISUAL_POINTS, stratify_col=None):
    if len(df) <= max_points:
        return df
    if stratify_col and stratify_col in df.columns and 1 < df[stratify_col].nunique(dropna=False) <= 50:
        parts = []
        for _, group in df.groupby(stratify_col, dropna=False, observed=True):
            allocation = max(1, round(len(group) / len(df) * max_points))
            parts.append(group.sample(n=min(len(group), allocation), random_state=42))
        sampled = pd.concat(parts).sample(frac=1, random_state=42)
        return sampled.sample(n=max_points, random_state=42) if len(sampled) > max_points else sampled
    return df.sample(n=max_points, random_state=42)


def get_visual_sample(df, dataset_id, max_points=MAX_VISUAL_POINTS, stratify_col=None):
    cache_key = (dataset_id, max_points, stratify_col)
    if st.session_state.get("visual_sample_key") != cache_key:
        st.session_state["visual_sample"] = create_visual_sample(df, max_points, stratify_col)
        st.session_state["visual_sample_key"] = cache_key
    return st.session_state["visual_sample"]


def show_visual_sampling_notice(full_rows, displayed_rows):
    if displayed_rows < full_rows:
        st.caption(f"Display only: {displayed_rows:,} representative points from {full_rows:,} rows. Statistics and models continue to use all rows.")


def exact_histogram_figure(series, column_name, template, color, bins=40):
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    counts, edges = np.histogram(values, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]
    figure = go.Figure(go.Bar(x=centers, y=counts, width=widths, marker_color=color, opacity=0.75))
    figure.update_layout(title=f"Distribution of {column_name} (all rows)", xaxis_title=column_name, yaxis_title="Probability density", template=template)
    return figure


def exact_density_figure(df, x_column, y_column, template, color_scale, contour=False):
    clean = df[[x_column, y_column]].apply(pd.to_numeric, errors="coerce").dropna()
    density, x_edges, y_edges = np.histogram2d(clean[x_column].to_numpy(), clean[y_column].to_numpy(), bins=30)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    trace_class = go.Contour if contour else go.Heatmap
    figure = go.Figure(trace_class(x=x_centers, y=y_centers, z=density.T, colorscale=color_scale, colorbar={"title": "Rows"}))
    figure.update_layout(title=f"Density: {x_column} vs {y_column} (all rows)", xaxis_title=x_column, yaxis_title=y_column, template=template)
    return figure


def aggregate_line_display(df, x_column, y_columns, color_column=None, max_points=MAX_VISUAL_POINTS):
    required = [x_column] + list(y_columns) + ([color_column] if color_column else [])
    working = df[required].dropna(subset=[x_column]).sort_values(x_column).copy()
    if len(working) <= max_points:
        return working
    grouped_frames = []
    groups = working.groupby(color_column, dropna=False, observed=True) if color_column else [(None, working)]
    for group_name, group in groups:
        allocation = max(1, round(len(group) / len(working) * max_points))
        bucket = np.minimum(np.floor(np.arange(len(group)) * allocation / len(group)).astype(int), allocation - 1)
        group = group.assign(_display_bucket=bucket)
        aggregation = {x_column: "first", **{column: "mean" for column in y_columns}}
        reduced = group.groupby("_display_bucket", as_index=False).agg(aggregation)
        if color_column:
            reduced[color_column] = group_name
        grouped_frames.append(reduced)
    return pd.concat(grouped_frames, ignore_index=True)

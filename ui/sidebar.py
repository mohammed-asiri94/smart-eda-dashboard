"""Sidebar controls and uploaded-dataset status."""

import streamlit as st


def render_dataset_sidebar(
    uploaded_file,
    selected_sheet,
    total_rows,
    total_columns,
    numeric_count,
    categorical_count,
    is_large,
    large_data_threshold,
):
    """Render dataset metadata and return whether the user requested a reset."""
    st.sidebar.markdown("### Uploaded file")
    st.sidebar.write(f"**Name:** {uploaded_file.name}")
    if selected_sheet:
        st.sidebar.write(f"**Sheet:** {selected_sheet}")
    st.sidebar.write(f"**Rows:** {total_rows:,}")
    st.sidebar.write(f"**Columns:** {total_columns:,}")
    st.sidebar.write(f"**Numeric:** {numeric_count}")
    st.sidebar.write(f"**Categorical:** {categorical_count}")
    if is_large:
        st.sidebar.success(
            f"Large Data Mode (full {total_rows:,} rows; disk-backed Parquet)"
        )
    else:
        st.sidebar.caption(
            f"Standard Mode (Large Data Mode starts at {large_data_threshold:,} rows)"
        )
    st.sidebar.markdown("---")
    return st.sidebar.button(
        "🔄 Start Over",
        use_container_width=True,
        help="Clear everything and remove the uploaded file.",
    )

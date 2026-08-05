"""Downloadable analysis reports page."""

from io import BytesIO

import pandas as pd
import streamlit as st

from modules.reports import generate_html_report, make_download_csv


def render_reports_page(
    df,
    uploaded_file,
    selected_sheet,
    total_rows,
    total_columns,
    total_missing_cells,
    missing_cells_pct,
    duplicate_rows,
    numeric_cols,
    categorical_cols,
    possible_id_cols,
    high_cardinality_cols,
    constant_cols,
    date_like_cols,
    outlier_cols_count,
    missing_report,
    dtype_report,
    unique_report,
    outlier_report,
    smart_recommendations,
):
    """Render report downloads from the current analysis state."""
    st.markdown("## Download Reports")

    html = generate_html_report(
        uploaded_file_name=uploaded_file.name,
        selected_sheet=selected_sheet,
        total_rows=total_rows,
        total_columns=total_columns,
        total_missing_cells=total_missing_cells,
        missing_cells_percentage=missing_cells_pct,
        duplicate_rows=duplicate_rows,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        possible_id_cols=possible_id_cols,
        high_cardinality_cols=high_cardinality_cols,
        constant_cols=constant_cols,
        date_like_cols=date_like_cols,
        outlier_columns_count=outlier_cols_count,
        missing_report=missing_report,
        dtype_report=dtype_report,
        unique_report=unique_report,
        outlier_report=outlier_report,
        smart_recommendations=smart_recommendations,
    )

    st.markdown("### Full HTML Report")
    st.download_button(
        "📥 Download Full EDA Report (HTML)",
        data=html, file_name="eda_report.html",
        mime="text/html", use_container_width=True,
    )

    st.markdown("---")

    # Excel Export
    st.markdown("### Export All Reports as Excel")
    if st.button("Generate Excel Report", key="excel_export", use_container_width=True):
        try:
            from io import BytesIO
            import openpyxl
            excel_buf = BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                missing_report.to_excel(writer, sheet_name="Missing Values", index=False)
                dtype_report.to_excel(writer, sheet_name="Column Types", index=False)
                unique_report.to_excel(writer, sheet_name="Unique Values", index=False)
                smart_recommendations.to_excel(writer, sheet_name="Recommendations", index=False)
                if not outlier_report.empty:
                    outlier_report.to_excel(writer, sheet_name="Outliers", index=False)
                summary_df2 = pd.DataFrame({
                    "Metric": ["Rows","Columns","Missing Cells","Missing %",
                               "Duplicate Rows","Numeric Columns","Categorical Columns"],
                    "Value":  [total_rows, total_columns, total_missing_cells,
                               missing_cells_pct, duplicate_rows,
                               len(numeric_cols), len(categorical_cols)],
                })
                summary_df2.to_excel(writer, sheet_name="Summary", index=False)
            excel_buf.seek(0)
            st.download_button(
                "Download Excel Report",
                data=excel_buf,
                file_name="eda_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as e:
            st.error("Excel export failed: " + str(e))

    st.markdown("---")
    st.markdown("### Smart Recommendations")
    make_download_csv(smart_recommendations, "recommendations.csv",
                      "📥 Download Recommendations")

    st.markdown("---")
    rr1, rr2, rr3 = st.columns(3)
    with rr1:
        st.markdown("### Column Types")
        make_download_csv(dtype_report, "column_types.csv", "📥 Download")
    with rr2:
        st.markdown("### Missing Values")
        make_download_csv(missing_report, "missing_values.csv", "📥 Download")
    with rr3:
        st.markdown("### Unique Values")
        make_download_csv(unique_report, "unique_values.csv", "📥 Download")

    rr4, rr5, rr6 = st.columns(3)
    with rr4:
        st.markdown("### Outliers")
        if not outlier_report.empty:
            make_download_csv(outlier_report, "outlier_report.csv", "📥 Download")
        else:
            st.info("No outlier report.")
    with rr5:
        st.markdown("### Original Data")
        st.download_button(
            "📥 Download original data (CSV)",
            data=df.to_csv(index=False).encode(),
            file_name="original_data.csv", mime="text/csv",
            use_container_width=True,
        )
    with rr6:
        st.markdown("### Summary")
        summary_df = pd.DataFrame({
            "Metric": [
                "File Name", "Excel Sheet", "Rows", "Columns",
                "Missing Cells", "Missing %", "Duplicate Rows",
                "Numeric Columns", "Categorical Columns",
                "Possible ID Columns", "High-Cardinality Columns",
                "Constant Columns", "Date-like Columns", "Outlier Columns",
            ],
            "Value": [
                uploaded_file.name,
                selected_sheet if selected_sheet else "N/A",
                total_rows, total_columns, total_missing_cells,
                missing_cells_pct, duplicate_rows,
                len(numeric_cols), len(categorical_cols),
                len(possible_id_cols), len(high_cardinality_cols),
                len(constant_cols), len(date_like_cols), outlier_cols_count,
            ],
        })
        summary_df["Value"] = summary_df["Value"].astype(str)
        make_download_csv(summary_df, "summary.csv", "📥 Download")

    st.markdown("### Summary Preview")
    st.dataframe(summary_df, use_container_width=True)

# ════════════════════════════════════════════════════
# Tab 8 — Raw Data
# ════════════════════════════════════════════════════


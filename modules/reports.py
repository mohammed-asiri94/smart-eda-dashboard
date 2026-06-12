# ============================================================
# modules/reports.py
# HTML report generation and CSV download helpers
# ============================================================

import streamlit as st
import pandas as pd


def make_download_csv(dataframe, file_name, label):
    st.download_button(
        label=label,
        data=dataframe.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )


def generate_html_report(
    uploaded_file_name,
    selected_sheet,
    total_rows,
    total_columns,
    total_missing_cells,
    missing_cells_percentage,
    duplicate_rows,
    numeric_cols,
    categorical_cols,
    possible_id_cols,
    high_cardinality_cols,
    constant_cols,
    date_like_cols,
    outlier_columns_count,
    missing_report,
    dtype_report,
    unique_report,
    outlier_report,
    smart_recommendations,
):
    outlier_html = (
        outlier_report.to_html(index=False)
        if not outlier_report.empty
        else "<p>No numeric columns available for outlier detection.</p>"
    )

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Smart EDA Report</title>
<style>
  body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    background: #F8FAFC;
    color: #0F172A;
  }}
  .header {{
    background: linear-gradient(135deg, #1E3A8A, #2563EB, #38BDF8);
    color: white;
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 30px;
  }}
  .header h1 {{ margin: 0; font-size: 36px; }}
  .header p {{ margin-top: 8px; color: #E0F2FE; }}
  .card {{
    background: white;
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 25px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 12px rgba(15,23,42,.05);
  }}
  h2 {{ color: #1E3A8A; border-bottom: 2px solid #E2E8F0; padding-bottom: 8px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 15px; font-size: 14px; }}
  th, td {{ border: 1px solid #CBD5E1; padding: 8px; text-align: left; }}
  th {{ background: #E0F2FE; }}
  .metrics {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
    margin-top: 20px;
  }}
  .metric {{
    background: #fff;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
  }}
  .metric-value {{ font-size: 24px; font-weight: bold; color: #2563EB; }}
  .metric-label {{ font-size: 13px; color: #64748B; margin-top: 5px; }}
</style>
</head>
<body>
<div class="header">
  <h1>Smart EDA Report</h1>
  <p>Automatic exploratory data analysis and data quality summary</p>
</div>

<div class="card">
  <h2>Dataset Summary</h2>
  <p><strong>File:</strong> {uploaded_file_name}</p>
  <p><strong>Sheet:</strong> {selected_sheet if selected_sheet else "N/A"}</p>
  <div class="metrics">
    <div class="metric"><div class="metric-value">{total_rows:,}</div><div class="metric-label">Rows</div></div>
    <div class="metric"><div class="metric-value">{total_columns:,}</div><div class="metric-label">Columns</div></div>
    <div class="metric"><div class="metric-value">{total_missing_cells:,}</div><div class="metric-label">Missing Cells</div></div>
    <div class="metric"><div class="metric-value">{missing_cells_percentage}%</div><div class="metric-label">Missing %</div></div>
    <div class="metric"><div class="metric-value">{duplicate_rows:,}</div><div class="metric-label">Duplicate Rows</div></div>
    <div class="metric"><div class="metric-value">{len(numeric_cols)}</div><div class="metric-label">Numeric Cols</div></div>
    <div class="metric"><div class="metric-value">{len(categorical_cols)}</div><div class="metric-label">Categorical Cols</div></div>
    <div class="metric"><div class="metric-value">{outlier_columns_count}</div><div class="metric-label">Outlier Cols</div></div>
  </div>
</div>

<div class="card">
  <h2>Smart Recommendations</h2>
  {smart_recommendations.to_html(index=False)}
</div>

<div class="card">
  <h2>Column Information</h2>
  {dtype_report.to_html(index=False)}
</div>

<div class="card">
  <h2>Missing Values</h2>
  {missing_report.to_html(index=False)}
</div>

<div class="card">
  <h2>Unique Values</h2>
  {unique_report.to_html(index=False)}
</div>

<div class="card">
  <h2>Outlier Report (IQR)</h2>
  {outlier_html}
</div>

<div class="card">
  <h2>Structural Checks</h2>
  <p><strong>Possible ID columns:</strong> {", ".join(possible_id_cols) if possible_id_cols else "None"}</p>
  <p><strong>High-cardinality columns:</strong> {", ".join(high_cardinality_cols) if high_cardinality_cols else "None"}</p>
  <p><strong>Constant columns:</strong> {", ".join(constant_cols) if constant_cols else "None"}</p>
  <p><strong>Date-like columns:</strong> {", ".join(date_like_cols) if date_like_cols else "None"}</p>
</div>
</body>
</html>
"""

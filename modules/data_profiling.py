# ============================================================
# modules/data_profiling.py
# Deep column profiling - numeric and categorical
# Skewness, kurtosis, distribution, rare categories
# ASCII only - no special characters
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# Numeric column profiling
# ============================================================

def _profile_numeric(series, col_name, plot_template):
    """Full profile for one numeric column."""
    s = series.dropna()
    n = len(s)
    n_missing = series.isnull().sum()

    if n == 0:
        st.warning("No valid values in " + col_name)
        return

    skewness = s.skew()
    kurtosis = s.kurtosis()
    cv = (s.std() / s.mean() * 100) if s.mean() != 0 else None

    # ── Metrics row ───────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("N",        n)
    c2.metric("Missing",  n_missing)
    c3.metric("Mean",     round(s.mean(), 3))
    c4.metric("Std Dev",  round(s.std(), 3))
    c5.metric("Skewness", round(skewness, 3))
    c6.metric("Kurtosis", round(kurtosis, 3))

    c7, c8, c9, c10, c11, c12 = st.columns(6)
    c7.metric("Min",    round(s.min(), 3))
    c8.metric("Q1",     round(s.quantile(0.25), 3))
    c9.metric("Median", round(s.median(), 3))
    c10.metric("Q3",    round(s.quantile(0.75), 3))
    c11.metric("Max",   round(s.max(), 3))
    c12.metric("CV %",  round(cv, 1) if cv else "N/A")

    # ── Distribution plot ─────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = px.histogram(
            s, nbins=30, marginal="box",
            title="Distribution of " + col_name,
            template=plot_template,
            color_discrete_sequence=["#2563EB"],
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        osm, osr = stats.probplot(s)[0]
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=list(osm), y=list(osr),
            mode="markers", marker=dict(size=4, color="#2563EB"),
            name="Data",
        ))
        fig_qq.add_trace(go.Scatter(
            x=[min(osm), max(osm)], y=[min(osm), max(osm)],
            mode="lines", line=dict(color="red", dash="dash"),
            name="Normal line",
        ))
        fig_qq.update_layout(
            title="Q-Q Plot (Normality Check)",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            template=plot_template,
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    # ── Normality test ────────────────────────────────────────
    if n <= 5000:
        sw_stat, sw_p = stats.shapiro(s)
        is_normal = sw_p >= 0.05
        st.caption(
            "Shapiro-Wilk: W=" + str(round(sw_stat, 4)) +
            ", p=" + str(round(sw_p, 4)) +
            (" -- Approximately normal" if is_normal else " -- Not normal")
        )

    # ── Automatic recommendations ─────────────────────────────
    recs = []

    if abs(skewness) > 1:
        direction = "right (positive)" if skewness > 0 else "left (negative)"
        recs.append({
            "Issue": "High skewness (" + str(round(skewness, 2)) + " - " + direction + ")",
            "Recommendation": "Consider log-transform (if all positive) or square-root transform",
            "Priority": "High",
        })
    elif abs(skewness) > 0.5:
        recs.append({
            "Issue": "Moderate skewness (" + str(round(skewness, 2)) + ")",
            "Recommendation": "May need transformation for parametric tests",
            "Priority": "Medium",
        })

    if abs(kurtosis) > 3:
        recs.append({
            "Issue": "High kurtosis (" + str(round(kurtosis, 2)) + ")",
            "Recommendation": "Heavy tails - outliers may be present",
            "Priority": "Medium",
        })

    if n_missing > 0:
        pct = round(n_missing / len(series) * 100, 1)
        recs.append({
            "Issue": str(pct) + "% missing values",
            "Recommendation": (
                "Low (<5%): remove rows or impute. "
                "Moderate (5-30%): impute with median/KNN. "
                "High (>30%): consider dropping column."
            ),
            "Priority": "High" if pct > 30 else ("Medium" if pct > 5 else "Low"),
        })

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    n_outliers = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
    if n_outliers > 0:
        recs.append({
            "Issue": str(n_outliers) + " outliers detected (IQR method)",
            "Recommendation": "Check if outliers are data errors or true extreme values",
            "Priority": "Medium",
        })

    if recs:
        st.markdown("**Recommendations:**")
        rec_df = pd.DataFrame(recs)
        st.dataframe(rec_df, use_container_width=True)
    else:
        st.success("No issues detected for this column.")


# ============================================================
# Categorical column profiling
# ============================================================

def _profile_categorical(series, col_name, plot_template):
    """Full profile for one categorical column."""
    s = series.dropna()
    n = len(s)
    n_missing  = series.isnull().sum()
    n_unique   = s.nunique()
    mode_val   = s.mode().iloc[0] if len(s) > 0 else "N/A"
    mode_freq  = (s == mode_val).sum()
    mode_pct   = round(mode_freq / len(s) * 100, 1) if len(s) > 0 else 0

    # Entropy (diversity measure)
    freqs = s.value_counts(normalize=True)
    entropy = -sum(p * np.log2(p) for p in freqs if p > 0)
    max_entropy = np.log2(n_unique) if n_unique > 1 else 1
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # ── Metrics ───────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("N",           n)
    c2.metric("Missing",     n_missing)
    c3.metric("Unique",      n_unique)
    c4.metric("Mode",        str(mode_val)[:15])
    c5.metric("Mode %",      str(mode_pct) + "%")

    c6, c7 = st.columns(2)
    c6.metric("Entropy (diversity)", round(entropy, 3),
              help="Higher = more diverse. Max = log2(unique categories)")
    c7.metric("Normalized entropy",  round(normalized_entropy, 3),
              help="0 = one dominant category, 1 = perfectly uniform")

    # ── Value counts ──────────────────────────────────────────
    vc = s.value_counts().reset_index()
    vc.columns = [col_name, "Count"]
    vc["Percentage"] = (vc["Count"] / len(s) * 100).round(2)
    vc["Cumulative %"] = vc["Percentage"].cumsum().round(2)

    top_n = min(20, len(vc))

    col_a, col_b = st.columns(2)
    with col_a:
        fig_bar = px.bar(
            vc.head(top_n),
            x="Count", y=col_name,
            orientation="h",
            title="Top " + str(top_n) + " Categories",
            template=plot_template,
            color="Count",
            color_continuous_scale="Blues",
        )
        fig_bar.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        if n_unique <= 10:
            fig_pie = px.pie(
                vc.head(top_n),
                names=col_name, values="Count",
                title="Category Distribution",
                template=plot_template,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            fig_bar2 = px.bar(
                vc.head(top_n),
                x=col_name, y="Percentage",
                title="Category % (top " + str(top_n) + ")",
                template=plot_template,
                color="Percentage",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig_bar2, use_container_width=True)

    # Full value counts table
    with st.expander("Full value counts table"):
        st.dataframe(vc, use_container_width=True)

    # ── Rare categories ───────────────────────────────────────
    rare_threshold = 0.05  # 5%
    rare = vc[vc["Percentage"] < rare_threshold * 100]
    n_rare = len(rare)

    # ── Recommendations ───────────────────────────────────────
    recs = []

    if n_missing > 0:
        pct = round(n_missing / len(series) * 100, 1)
        recs.append({
            "Issue": str(pct) + "% missing values",
            "Recommendation": "Impute with mode or create 'Unknown' category",
            "Priority": "High" if pct > 30 else ("Medium" if pct > 5 else "Low"),
        })

    if n_rare > 0:
        recs.append({
            "Issue": str(n_rare) + " rare categories (< 5% each): " +
                     ", ".join(rare[col_name].astype(str).tolist()[:5]),
            "Recommendation": "Consider merging rare categories into 'Other' to avoid sparse cells",
            "Priority": "Medium",
        })

    if n_unique > 50:
        recs.append({
            "Issue": "High cardinality (" + str(n_unique) + " unique values)",
            "Recommendation": "Use target encoding or group categories before modeling",
            "Priority": "Medium",
        })

    if mode_pct > 90:
        recs.append({
            "Issue": "Dominant category (" + str(mode_val) + " = " + str(mode_pct) + "%)",
            "Recommendation": "Nearly constant variable - may not be useful for modeling",
            "Priority": "High",
        })

    if recs:
        st.markdown("**Recommendations:**")
        st.dataframe(pd.DataFrame(recs), use_container_width=True)
    else:
        st.success("No issues detected for this column.")


# ============================================================
# Summary profiling table (all columns)
# ============================================================

@st.cache_data(show_spinner=False)
def _build_summary_table(df):
    """Build a summary table for all columns."""
    rows = []
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols     = df.select_dtypes(include=["object","category","bool"]).columns.tolist()

    for col in df.columns:
        s = df[col]
        n_missing = s.isnull().sum()
        pct_missing = round(n_missing / len(s) * 100, 2)
        n_unique  = s.nunique(dropna=True)

        row = {
            "Column":       col,
            "Type":         str(s.dtype),
            "N valid":      int(s.notna().sum()),
            "Missing %":    pct_missing,
            "Unique":       n_unique,
        }

        if col in numeric_cols:
            s_clean = s.dropna()
            row["Mean"]     = round(s_clean.mean(), 3) if len(s_clean) > 0 else None
            row["Std"]      = round(s_clean.std(), 3)  if len(s_clean) > 0 else None
            row["Skewness"] = round(s_clean.skew(), 3) if len(s_clean) > 0 else None
            row["Min"]      = round(s_clean.min(), 3)  if len(s_clean) > 0 else None
            row["Max"]      = round(s_clean.max(), 3)  if len(s_clean) > 0 else None
            row["Needs transform"] = "YES" if len(s_clean) > 0 and abs(s_clean.skew()) > 1 else "NO"
        else:
            mode_vals = s.mode(dropna=True)
            row["Mode"] = str(mode_vals.iloc[0])[:20] if len(mode_vals) > 0 else "N/A"
            row["Needs transform"] = "N/A"

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Main render function
# ============================================================

def render_data_profiling(df, plot_template):
    st.markdown("## Data Profiling")
    st.info(
        "Deep analysis of each column — distribution, skewness, outliers, "
        "rare categories, and automatic recommendations."
    )

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols     = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # ── Summary table ─────────────────────────────────────────
    st.markdown("### Column Summary Table")
    summary_df = _build_summary_table(df)
    st.dataframe(summary_df, use_container_width=True)

    st.download_button(
        "Download profiling summary (CSV)",
        data=summary_df.to_csv(index=False).encode(),
        file_name="data_profiling_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("---")

    # ── Column selector ───────────────────────────────────────
    st.markdown("### Deep Profile — Select Column")

    profile_type = st.radio(
        "Column type",
        ["Numeric columns", "Categorical columns"],
        horizontal=True,
        key="prof_type",
    )

    if profile_type == "Numeric columns":
        if not numeric_cols:
            st.info("No numeric columns.")
            return
        selected = st.selectbox(
            "Select numeric column",
            numeric_cols,
            key="prof_num_col",
        )
        st.markdown("### Profile: " + selected)
        _profile_numeric(df[selected], selected, plot_template)

        # Quick browse all
        if st.checkbox("Browse all numeric columns", key="prof_browse_num"):
            for col in numeric_cols:
                if col != selected:
                    with st.expander("Column: " + col):
                        _profile_numeric(df[col], col, plot_template)

    else:
        if not cat_cols:
            st.info("No categorical columns.")
            return
        selected = st.selectbox(
            "Select categorical column",
            cat_cols,
            key="prof_cat_col",
        )
        st.markdown("### Profile: " + selected)
        _profile_categorical(df[selected], selected, plot_template)

        if st.checkbox("Browse all categorical columns", key="prof_browse_cat"):
            for col in cat_cols:
                if col != selected:
                    with st.expander("Column: " + col):
                        _profile_categorical(df[col], col, plot_template)

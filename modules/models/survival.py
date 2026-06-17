# ============================================================
# modules/models/survival.py
# Survival Analysis — Kaplan-Meier + Cox Proportional Hazards
# Full diagnostics + fix suggestions on the same page
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import concordance_index
from scipy import stats


# ============================================================
# Helpers
# ============================================================

def _check_columns(df, duration_col, event_col):
    """Validate that duration and event columns are usable."""
    issues = []
    if df[duration_col].isnull().any():
        issues.append(f"Duration column '{duration_col}' has missing values — rows will be dropped.")
    if df[event_col].isnull().any():
        issues.append(f"Event column '{event_col}' has missing values — rows will be dropped.")
    if not set(df[event_col].dropna().unique()).issubset({0, 1, True, False}):
        issues.append(
            f"Event column '{event_col}' should be binary (0/1). "
            f"Found: {sorted(df[event_col].dropna().unique())}"
        )
    if (df[duration_col].dropna() < 0).any():
        issues.append(f"Duration column '{duration_col}' contains negative values.")
    return issues


def _prep_survival_data(df, duration_col, event_col, covariate_cols=None):
    cols = [duration_col, event_col] + (covariate_cols or [])
    data = df[cols].dropna().copy()
    data[event_col] = data[event_col].astype(int)
    return data


def _show_issue(level, msg, fix=""):
    if level == "error":
        st.error(f"🔴 **{msg}**" + (f"\n\n💡 *Fix:* {fix}" if fix else ""))
    elif level == "warning":
        st.warning(f"🟡 **{msg}**" + (f"\n\n💡 *Fix:* {fix}" if fix else ""))
    else:
        st.info(f"🔵 {msg}")


def _diagnostic_summary(issues):
    if not issues:
        st.success("✅ All diagnostic checks passed — no major issues detected.")
    else:
        for i in issues:
            _show_issue(i["level"], i["msg"], i.get("fix", ""))


# ============================================================
# Kaplan-Meier
# ============================================================

def run_kaplan_meier(df, duration_col, event_col, group_col, plot_template):
    st.markdown("## Kaplan-Meier Survival Analysis")

    # Validate
    col_issues = _check_columns(df, duration_col, event_col)
    for msg in col_issues:
        st.warning(f"⚠️ {msg}")

    data = _prep_survival_data(df, duration_col, event_col)
    if data.empty:
        st.error("No data remaining after dropping missing values.")
        return

    n_total    = len(data)
    n_events   = int(data[event_col].sum())
    n_censored = n_total - n_events
    pct_events = round(n_events / n_total * 100, 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total observations", f"{n_total:,}")
    c2.metric("Events (uncensored)", f"{n_events:,}")
    c3.metric("Censored", f"{n_censored:,}")
    c4.metric("Event rate", f"{pct_events}%")

    if n_events < 10:
        st.warning(
            "⚠️ Fewer than 10 events detected. "
            "KM estimates will be unstable — interpret with caution."
        )

    # ── KM curve ─────────────────────────────────────────────
    st.markdown("### Survival Curve")

    fig = go.Figure()

    if group_col and group_col != "None":
        # Check if group column exists and has valid groups
        if group_col not in data.columns:
            data2 = _prep_survival_data(df, duration_col, event_col)
        else:
            data2 = _prep_survival_data(df, duration_col, event_col, [group_col])

        groups = data2[group_col].unique()

        if len(groups) > 10:
            st.warning("Too many groups (>10). Showing overall curve only.")
            group_col = None
        else:
            colors = px.colors.qualitative.Set2
            group_results = {}

            for i, grp in enumerate(sorted(groups)):
                mask = data2[group_col] == grp
                grp_data = data2[mask]
                kmf = KaplanMeierFitter()
                kmf.fit(
                    grp_data[duration_col],
                    event_observed=grp_data[event_col],
                    label=str(grp),
                )
                group_results[grp] = kmf

                t = kmf.survival_function_.index.values
                s = kmf.survival_function_.iloc[:, 0].values
                ci_lower = kmf.confidence_interval_.iloc[:, 0].values
                ci_upper = kmf.confidence_interval_.iloc[:, 1].values
                color = colors[i % len(colors)]

                # CI band
                fig.add_trace(go.Scatter(
                    x=np.concatenate([t, t[::-1]]),
                    y=np.concatenate([ci_upper, ci_lower[::-1]]),
                    fill="toself",
                    fillcolor=color.replace("rgb", "rgba").replace(")", ",0.15)"),
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                # KM step
                fig.add_trace(go.Scatter(
                    x=t, y=s,
                    mode="lines",
                    name=f"{group_col}={grp} (n={mask.sum()})",
                    line=dict(color=color, width=2, shape="hv"),
                ))

            # Log-rank test
            if len(groups) == 2:
                grp_list = sorted(groups)
                d0 = data2[data2[group_col] == grp_list[0]]
                d1 = data2[data2[group_col] == grp_list[1]]
                lr = logrank_test(
                    d0[duration_col], d1[duration_col],
                    event_observed_A=d0[event_col],
                    event_observed_B=d1[event_col],
                )
                st.markdown("### Log-Rank Test")
                lr1, lr2, lr3 = st.columns(3)
                lr1.metric("Test statistic", round(lr.test_statistic, 4))
                lr2.metric("p-value", round(lr.p_value, 5))
                lr3.metric("Result",
                           "✅ Significant (p<0.05)" if lr.p_value < 0.05
                           else "❌ Not significant")
                if lr.p_value < 0.05:
                    st.success(
                        f"Survival curves differ significantly between groups "
                        f"(p = {lr.p_value:.4f})."
                    )
                else:
                    st.info(
                        f"No significant difference between groups "
                        f"(p = {lr.p_value:.4f})."
                    )

            elif len(groups) > 2:
                mlr = multivariate_logrank_test(
                    data2[duration_col],
                    data2[group_col],
                    event_col=data2[event_col],
                )
                st.markdown("### Multivariate Log-Rank Test")
                ml1, ml2 = st.columns(2)
                ml1.metric("Test statistic", round(mlr.test_statistic, 4))
                ml2.metric("p-value", round(mlr.p_value, 5))

    if not group_col or group_col == "None":
        kmf = KaplanMeierFitter()
        kmf.fit(data[duration_col], event_observed=data[event_col], label="Overall")
        t = kmf.survival_function_.index.values
        s = kmf.survival_function_.iloc[:, 0].values
        ci_lower = kmf.confidence_interval_.iloc[:, 0].values
        ci_upper = kmf.confidence_interval_.iloc[:, 1].values

        fig.add_trace(go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([ci_upper, ci_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(37,99,235,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=t, y=s,
            mode="lines", name="Overall",
            line=dict(color="#2563EB", width=2, shape="hv"),
        ))

        # Median survival
        med = kmf.median_survival_time_
        if med and not np.isinf(med):
            fig.add_hline(
                y=0.5, line_dash="dash", line_color="red",
                annotation_text=f"Median = {med:.2f}",
            )
            st.metric("Median Survival Time", round(med, 3))

    fig.update_layout(
        title="Kaplan-Meier Survival Curve",
        xaxis_title=f"Time ({duration_col})",
        yaxis_title="Survival Probability",
        yaxis=dict(range=[0, 1.05]),
        template=plot_template,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Survival table ────────────────────────────────────────
    st.markdown("### Survival Table")
    kmf_tbl = KaplanMeierFitter()
    kmf_tbl.fit(data[duration_col], event_observed=data[event_col])
    tbl = kmf_tbl.survival_function_.copy()
    tbl.index = tbl.index.round(3)
    tbl.columns = ["Survival Probability"]
    tbl["CI Lower"] = kmf_tbl.confidence_interval_.iloc[:, 0].round(4)
    tbl["CI Upper"] = kmf_tbl.confidence_interval_.iloc[:, 1].round(4)
    st.dataframe(tbl.reset_index().rename(columns={"index": "Time"}),
                 use_container_width=True)

    st.download_button(
        "📥 Download survival table (CSV)",
        data=tbl.reset_index().to_csv().encode(),
        file_name="km_survival_table.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Diagnostics ───────────────────────────────────────────
    st.markdown("### 🔬 Diagnostics")
    issues = []

    if pct_events < 10:
        issues.append({
            "level": "error",
            "msg": f"Very low event rate ({pct_events}%). KM estimates unreliable.",
            "fix": "Collect more data or extend follow-up period.",
        })
    elif pct_events < 30:
        issues.append({
            "level": "warning",
            "msg": f"Low event rate ({pct_events}%). Estimates may be imprecise.",
            "fix": "Interpret confidence intervals carefully.",
        })

    if n_censored / n_total > 0.70:
        issues.append({
            "level": "warning",
            "msg": f"High censoring rate ({round(n_censored/n_total*100,1)}%). "
                   "Assumes censoring is non-informative.",
            "fix": "Verify that censoring is random (non-informative). "
                   "If informative censoring is suspected, consider sensitivity analysis.",
        })

    st.markdown("### 🩺 Diagnostic Summary")
    _diagnostic_summary(issues)


# ============================================================
# Cox Proportional Hazards
# ============================================================

def run_cox_ph(df, duration_col, event_col, covariate_cols, plot_template):
    st.markdown("## Cox Proportional Hazards Model")

    col_issues = _check_columns(df, duration_col, event_col)
    for msg in col_issues:
        st.warning(f"⚠️ {msg}")

    data = _prep_survival_data(df, duration_col, event_col, covariate_cols)
    if data.empty:
        st.error("No data remaining after dropping missing values.")
        return

    n_total  = len(data)
    n_events = int(data[event_col].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Observations", f"{n_total:,}")
    c2.metric("Events", f"{n_events:,}")
    c3.metric("Event rate", f"{round(n_events/n_total*100,1)}%")

    # EPV check (Events Per Variable)
    epv = n_events / len(covariate_cols) if covariate_cols else 0
    if epv < 10:
        st.warning(
            f"⚠️ Events Per Variable (EPV) = {epv:.1f} — below the recommended minimum of 10. "
            "Model may be overfitted. Consider reducing the number of covariates."
        )

    # ── Fit Cox model ─────────────────────────────────────────
    try:
        cph = CoxPHFitter()
        cph.fit(
            data,
            duration_col=duration_col,
            event_col=event_col,
            show_progress=False,
        )
    except Exception as e:
        st.error(f"Cox model failed to converge: {e}")
        return

    st.success("✅ Cox PH model fitted successfully.")

    # ── Fit metrics ───────────────────────────────────────────
    c_idx = concordance_index(
        data[duration_col], -cph.predict_partial_hazard(data), data[event_col]
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Concordance Index (C)", round(c_idx, 4))
    m2.metric("Log-Likelihood", round(cph.log_likelihood_, 4))
    m3.metric("AIC", round(cph.AIC_, 4))

    if c_idx < 0.6:
        st.warning("⚠️ C-index < 0.6 — model has weak discriminative ability.")
    elif c_idx >= 0.7:
        st.success("✅ C-index ≥ 0.7 — model has good discriminative ability.")

    # ── Coefficients table ────────────────────────────────────
    st.markdown("### Coefficients — Hazard Ratios")
    summary = cph.summary.copy()
    summary = summary.reset_index()
    summary.columns = [c.replace(" ", "_") for c in summary.columns]

    coef_tbl = pd.DataFrame({
        "Variable":    summary["covariate"],
        "Coefficient": summary["coef"].round(4),
        "Std Error":   summary["se(coef)"].round(4),
        "Hazard Ratio (HR)": np.exp(summary["coef"]).round(4),
        "HR CI Lower": np.exp(summary["coef lower 95%"]).round(4),
        "HR CI Upper": np.exp(summary["coef upper 95%"]).round(4),
        "p-value":     summary["p"].round(5),
        "Significant": summary["p"].apply(lambda p: "✅" if p < 0.05 else ""),
    })
    st.dataframe(coef_tbl, use_container_width=True)

    st.download_button(
        "📥 Download coefficients (CSV)",
        data=coef_tbl.to_csv(index=False).encode(),
        file_name="cox_coefficients.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Forest plot ───────────────────────────────────────────
    st.markdown("### Forest Plot — Hazard Ratios")
    coef_tbl_sorted = coef_tbl.sort_values("Hazard Ratio (HR)", ascending=True)
    fig_forest = go.Figure()
    fig_forest.add_trace(go.Scatter(
        x=coef_tbl_sorted["Hazard Ratio (HR)"],
        y=coef_tbl_sorted["Variable"],
        mode="markers",
        marker=dict(size=10, color="#2563EB"),
        error_x=dict(
            type="data",
            symmetric=False,
            array=(coef_tbl_sorted["HR CI Upper"] -
                   coef_tbl_sorted["Hazard Ratio (HR)"]).values,
            arrayminus=(coef_tbl_sorted["Hazard Ratio (HR)"] -
                        coef_tbl_sorted["HR CI Lower"]).values,
            color="#2563EB",
        ),
        name="HR (95% CI)",
    ))
    fig_forest.add_vline(x=1, line_dash="dash", line_color="red",
                         annotation_text="HR = 1 (no effect)")
    fig_forest.update_layout(
        title="Hazard Ratio Forest Plot",
        xaxis_title="Hazard Ratio (HR)",
        template=plot_template,
        height=max(300, len(covariate_cols) * 40 + 100),
    )
    st.plotly_chart(fig_forest, use_container_width=True)

    # ── Baseline survival curve ───────────────────────────────
    st.markdown("### Baseline Survival Curve")
    baseline = cph.baseline_survival_
    fig_base = go.Figure()
    fig_base.add_trace(go.Scatter(
        x=baseline.index,
        y=baseline.iloc[:, 0],
        mode="lines",
        line=dict(color="#2563EB", width=2, shape="hv"),
        name="Baseline Survival",
    ))
    fig_base.update_layout(
        title="Baseline Survival Function",
        xaxis_title=f"Time ({duration_col})",
        yaxis_title="Survival Probability",
        yaxis=dict(range=[0, 1.05]),
        template=plot_template,
    )
    st.plotly_chart(fig_base, use_container_width=True)

    # ── Diagnostics ───────────────────────────────────────────
    st.markdown("### 🔬 Diagnostics")
    issues = []

    # ── 1. Proportional Hazards assumption — Schoenfeld residuals
    st.markdown("#### 1. Proportional Hazards Assumption (Schoenfeld Residuals)")
    st.caption(
        "Tests whether the hazard ratio is constant over time. "
        "p < 0.05 suggests the PH assumption is violated for that variable."
    )
    try:
        from lifelines.statistics import proportional_hazard_test
        ph_test = proportional_hazard_test(cph, data, time_transform="rank")
        ph_results = ph_test.summary.copy().reset_index()
        ph_results.columns = ["Variable", "Test Statistic", "p-value"]
        ph_results["PH Assumption"] = ph_results["p-value"].apply(
            lambda p: "✅ Met" if p >= 0.05 else "🔴 Violated"
        )
        ph_results = ph_results.round(5)
        st.dataframe(ph_results, use_container_width=True)

        violated = ph_results[ph_results["p-value"] < 0.05]["Variable"].tolist()
        if violated:
            issues.append({
                "level": "error",
                "msg": f"PH assumption violated for: {', '.join(violated)}",
                "fix": (
                    "Option 1: Use Stratified Cox — stratify by the violating variable.\n"
                    "Option 2: Add time-varying coefficients (interaction with time).\n"
                    "Option 3: Use an Accelerated Failure Time (AFT) model instead."
                ),
            })
        else:
            st.success("✅ Proportional Hazards assumption holds for all variables.")

    except Exception as e:
        st.info(f"PH test could not be performed: {e}")

    # ── 2. Martingale residuals plot
    st.markdown("#### 2. Martingale Residuals")
    st.caption(
        "Checks for non-linear relationships between covariates and the outcome. "
        "A smooth trend away from zero suggests non-linearity."
    )
    try:
        martingale = data[event_col] - cph.predict_cumulative_hazard(data).iloc[-1]

        num_covariates = [
            c for c in covariate_cols
            if pd.api.types.is_numeric_dtype(data[c])
        ]

        if num_covariates:
            sel_cov = st.selectbox(
                "Plot Martingale residuals vs covariate:",
                num_covariates,
                key="cox_mart_cov",
            )
            fig_mart = go.Figure()
            fig_mart.add_trace(go.Scatter(
                x=data[sel_cov],
                y=martingale,
                mode="markers",
                marker=dict(size=4, color="#2563EB", opacity=0.5),
                name="Martingale residuals",
            ))
            fig_mart.add_hline(y=0, line_dash="dash", line_color="red")
            fig_mart.update_layout(
                title=f"Martingale Residuals vs {sel_cov}",
                xaxis_title=sel_cov,
                yaxis_title="Martingale Residual",
                template=plot_template,
            )
            st.plotly_chart(fig_mart, use_container_width=True)
            st.caption(
                "If you see a clear curve (not random scatter around 0), "
                "consider adding a log or polynomial term for this variable."
            )
    except Exception:
        st.info("Martingale residuals could not be computed.")

    # ── 3. Log-Log plot (visual PH check)
    st.markdown("#### 3. Log-Log Plot (Visual PH Check)")
    st.caption(
        "If the log-log survival curves are parallel, "
        "the proportional hazards assumption holds."
    )
    try:
        if "group_col_for_loglog" in st.session_state:
            pass
        cat_cols_available = [
            c for c in covariate_cols
            if data[c].nunique() <= 6
        ]
        if cat_cols_available:
            loglog_col = st.selectbox(
                "Select grouping variable for log-log plot:",
                cat_cols_available,
                key="cox_loglog_col",
            )
            fig_ll = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, grp in enumerate(sorted(data[loglog_col].unique())):
                mask = data[loglog_col] == grp
                kmf_ll = KaplanMeierFitter()
                kmf_ll.fit(
                    data[mask][duration_col],
                    event_observed=data[mask][event_col],
                )
                t = kmf_ll.survival_function_.index.values[1:]
                s = kmf_ll.survival_function_.iloc[1:, 0].values
                with np.errstate(divide="ignore", invalid="ignore"):
                    ll = np.log(-np.log(np.where(s > 0, s, np.nan)))
                fig_ll.add_trace(go.Scatter(
                    x=np.log(t + 1e-9), y=ll,
                    mode="lines",
                    name=f"{loglog_col}={grp}",
                    line=dict(color=colors[i % len(colors)], width=2),
                ))
            fig_ll.update_layout(
                title="Log-Log Plot",
                xaxis_title="log(Time)",
                yaxis_title="log(-log(S(t)))",
                template=plot_template,
            )
            st.plotly_chart(fig_ll, use_container_width=True)
        else:
            st.info("No categorical covariate with ≤ 6 groups available for log-log plot.")
    except Exception:
        st.info("Log-log plot could not be generated.")

    # ── 4. EPV warning
    if epv < 10:
        issues.append({
            "level": "error",
            "msg": f"Events Per Variable (EPV) = {epv:.1f} — model may be overfitted.",
            "fix": "Reduce the number of covariates or collect more data.",
        })

    # ── 5. Concordance
    if c_idx < 0.55:
        issues.append({
            "level": "warning",
            "msg": f"C-index = {c_idx:.4f} — very low discriminative ability.",
            "fix": "Consider adding stronger predictors or interaction terms.",
        })

    # ── Full summary ──────────────────────────────────────────
    st.markdown("### 🩺 Diagnostic Summary")
    _diagnostic_summary(issues)

    with st.expander("Full Model Summary"):
        st.text(cph.summary.to_string())

    # ── Stratified Cox option (if PH violated) ────────────────
    if issues and any("PH assumption violated" in i["msg"] for i in issues):
        st.markdown("---")
        st.markdown("### 🔧 Fix: Stratified Cox Model")
        st.info(
            "Since PH assumption is violated, you can fit a Stratified Cox model "
            "which relaxes the PH assumption for the stratification variable."
        )

        strat_options = [
            c for c in covariate_cols
            if data[c].nunique() <= 10
        ]
        if strat_options:
            strat_col = st.selectbox(
                "Stratify by:", strat_options, key="cox_strat_col"
            )
            if st.button("▶ Fit Stratified Cox", key="cox_strat_btn"):
                try:
                    remaining_covs = [c for c in covariate_cols if c != strat_col]
                    if not remaining_covs:
                        st.warning("Need at least one covariate besides the stratification variable.")
                    else:
                        cph_strat = CoxPHFitter()
                        cph_strat.fit(
                            data,
                            duration_col=duration_col,
                            event_col=event_col,
                            strata=[strat_col],
                            show_progress=False,
                        )
                        st.success(f"✅ Stratified Cox fitted (stratified by '{strat_col}').")
                        strat_summary = cph_strat.summary.reset_index()
                        strat_summary.columns = [
                            c.replace(" ", "_") for c in strat_summary.columns
                        ]
                        strat_tbl = pd.DataFrame({
                            "Variable":    strat_summary["covariate"],
                            "Coefficient": strat_summary["coef"].round(4),
                            "HR":  np.exp(strat_summary["coef"]).round(4),
                            "HR CI Lower": np.exp(strat_summary["coef lower 95%"]).round(4),
                            "HR CI Upper": np.exp(strat_summary["coef upper 95%"]).round(4),
                            "p-value":     strat_summary["p"].round(5),
                        })
                        st.dataframe(strat_tbl, use_container_width=True)
                        c_strat = concordance_index(
                            data[duration_col],
                            -cph_strat.predict_partial_hazard(data),
                            data[event_col],
                        )
                        st.metric("Stratified C-index", round(c_strat, 4))
                except Exception as e:
                    st.error(f"Stratified Cox failed: {e}")


# ============================================================
# Main render function — called from app.py Tab
# ============================================================

def render_survival_tab(df, df_cleaned, plot_template):
    st.markdown("# 🫀 Survival Analysis")
    st.info(
        "Survival analysis models **time until an event** occurs, "
        "accounting for censored observations (subjects who did not experience "
        "the event during follow-up)."
    )

    # Dataset choice
    dataset_choice = st.radio(
        "Dataset to use",
        ["Original data", "Cleaned data (from Data Cleaning tab)"],
        horizontal=True,
        key="surv_dataset",
    )
    mdf = df_cleaned.copy() if dataset_choice.startswith("Cleaned") else df.copy()

    all_cols     = mdf.columns.tolist()
    numeric_cols = mdf.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns (duration + event).")
        return

    # ── Column selection ──────────────────────────────────────
    st.markdown("### Column Setup")
    s1, s2, s3 = st.columns(3)

    with s1:
        duration_col = st.selectbox(
            "⏱ Duration column (time to event or censoring)",
            numeric_cols,
            key="surv_duration",
            help="Must be numeric and non-negative.",
        )
    with s2:
        event_options = [c for c in all_cols if c != duration_col]
        event_col = st.selectbox(
            "🎯 Event column (1 = event occurred, 0 = censored)",
            event_options,
            key="surv_event",
            help="Must be binary: 1 = event, 0 = censored.",
        )

        # ── Immediate readiness check ────────────────────────
        unique_event_vals = mdf[event_col].dropna().unique()
        if not set(unique_event_vals).issubset({0, 1, True, False}):
            st.warning(
                "This column has values " + str(sorted(unique_event_vals)[:5]) +
                " — Event column should be binary (0/1). "
                "Choose a different column or recode this one first."
            )

        if (mdf[duration_col].dropna() < 0).any():
            st.warning(
                "Duration column '" + duration_col + "' contains negative values, "
                "which is not valid for survival analysis."
            )
    with s3:
        group_options = ["None"] + [c for c in all_cols
                                    if c not in (duration_col, event_col)]
        group_col = st.selectbox(
            "👥 Grouping column (optional, for KM stratification)",
            group_options,
            key="surv_group",
        )

    # ── Model selection ───────────────────────────────────────
    st.markdown("### Model Selection")
    model_type = st.radio(
        "Choose model",
        ["Kaplan-Meier", "Cox Proportional Hazards"],
        horizontal=True,
        key="surv_model_type",
    )

    covariate_cols = []
    if model_type == "Cox Proportional Hazards":
        available_covs = [
            c for c in all_cols
            if c not in (duration_col, event_col)
        ]
        covariate_cols = st.multiselect(
            "Covariates (predictors)",
            available_covs,
            default=available_covs[:min(5, len(available_covs))],
            key="surv_covariates",
        )

    # ── Guidance ──────────────────────────────────────────────
    with st.expander("📖 When to use each model"):
        st.markdown("""
**Kaplan-Meier:**
- Non-parametric — no assumptions about the distribution of survival times
- Best for: visualizing and comparing survival curves between groups
- Limitation: cannot adjust for confounders

**Cox Proportional Hazards:**
- Semi-parametric — models the effect of covariates on the hazard rate
- Best for: identifying factors that influence survival time
- Key assumption: Proportional Hazards (HR is constant over time)
- Can handle: continuous and categorical covariates, time-to-event outcomes
        """)

    # ── Run button ────────────────────────────────────────────
    if st.button("▶ Run Survival Analysis", use_container_width=True, key="surv_run"):
        try:
            if model_type == "Kaplan-Meier":
                run_kaplan_meier(
                    mdf, duration_col, event_col,
                    None if group_col == "None" else group_col,
                    plot_template,
                )
            elif model_type == "Cox Proportional Hazards":
                if not covariate_cols:
                    st.error("Select at least one covariate for Cox PH.")
                else:
                    run_cox_ph(
                        mdf, duration_col, event_col,
                        covariate_cols, plot_template,
                    )
        except Exception as e:
            st.error(f"Survival analysis error: {e}")
            with st.expander("Error details"):
                import traceback
                st.code(traceback.format_exc())
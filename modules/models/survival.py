# ============================================================
# modules/models/survival.py
# Survival Analysis — Kaplan-Meier, Nelson-Aalen, Cox PH,
# Stratified Cox, and Parametric AFT models
# Full diagnostics + practical warnings/explanations
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from scipy import stats

from lifelines import (
    KaplanMeierFitter,
    NelsonAalenFitter,
    CoxPHFitter,
    WeibullAFTFitter,
    LogNormalAFTFitter,
    LogLogisticAFTFitter,
)
from lifelines.statistics import logrank_test, multivariate_logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index


# ============================================================
# Shared helpers
# ============================================================

def _show_issue(level, msg, fix=""):
    if level == "error":
        st.error("🔴 **" + msg + "**" + ("\n\n💡 *Suggested action:* " + fix if fix else ""))
    elif level == "warning":
        st.warning("🟡 **" + msg + "**" + ("\n\n💡 *Suggested action:* " + fix if fix else ""))
    else:
        st.info("🔵 " + msg + ("\n\n💡 *Tip:* " + fix if fix else ""))


def _diagnostic_summary(issues):
    if not issues:
        st.success("✅ No major diagnostic issues detected.")
    else:
        for issue in issues:
            _show_issue(issue.get("level", "warning"), issue.get("msg", ""), issue.get("fix", ""))


def _safe_internal_name(prefix, base, existing_cols):
    safe = "".join(ch if str(ch).isalnum() else "_" for ch in str(base)).strip("_") or "var"
    name = f"__{prefix}_{safe}__"
    i = 1
    while name in existing_cols:
        name = f"__{prefix}_{safe}_{i}__"
        i += 1
    return name


def _make_event_binary(df, event_col, event_value, output_col="__survival_event_binary__"):
    """Create an internal lifelines-ready event indicator from any user-selected event code.

    User-facing UI should describe values as Event occurred / Censored.
    Internally, lifelines still requires 1 for event and 0 for censored.
    """
    out = df.copy()
    existing = set(out.columns)
    if output_col in existing:
        output_col = _safe_internal_name("survival_event", event_col, existing)

    raw = out[event_col]
    if pd.api.types.is_bool_dtype(raw):
        out[output_col] = raw.astype(int)
    else:
        # Compare as strings to support numeric, text, yes/no, dead/alive, etc.
        out[output_col] = raw.astype(str).eq(str(event_value)).astype(int)
    return out, output_col


def _check_survival_inputs(df, duration_col, event_col):
    issues = []
    if duration_col not in df.columns or event_col not in df.columns:
        issues.append({"level": "error", "msg": "Selected duration/event columns are not in the dataset."})
        return issues
    if df[duration_col].isnull().any():
        issues.append({
            "level": "warning",
            "msg": f"Duration column '{duration_col}' contains missing values.",
            "fix": "Rows with missing duration/event data will be excluded from this survival analysis.",
        })
    if df[event_col].isnull().any():
        issues.append({
            "level": "warning",
            "msg": f"Event column '{event_col}' contains missing values.",
            "fix": "Rows with missing duration/event data will be excluded from this survival analysis.",
        })
    try:
        if (pd.to_numeric(df[duration_col], errors="coerce").dropna() < 0).any():
            issues.append({
                "level": "error",
                "msg": f"Duration column '{duration_col}' contains negative values.",
                "fix": "Survival time must be non-negative. Check coding or remove invalid records.",
            })
    except Exception:
        issues.append({
            "level": "error",
            "msg": f"Duration column '{duration_col}' must be numeric.",
            "fix": "Choose a numeric time-to-event column.",
        })
    return issues


def _prep_survival_data(df, duration_col, event_col, extra_cols=None):
    cols = [duration_col, event_col] + (extra_cols or [])
    cols = list(dict.fromkeys([c for c in cols if c in df.columns]))
    data = df[cols].copy()
    data[duration_col] = pd.to_numeric(data[duration_col], errors="coerce")
    data[event_col] = pd.to_numeric(data[event_col], errors="coerce")
    data = data.dropna(subset=[duration_col, event_col]).copy()
    data[event_col] = data[event_col].astype(int)
    data = data[data[duration_col] >= 0].copy()
    return data


def _format_p(p):
    if p is None or pd.isna(p):
        return "N/A"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"




def _display_value(col, value, label_map=None):
    """Return a user-facing label for a raw value when optional value labels are supplied."""
    label_map = label_map or {}
    col_map = label_map.get(col, {}) if isinstance(label_map, dict) else {}
    key = str(value)
    if key in col_map and str(col_map[key]).strip():
        return str(col_map[key]).strip()
    return str(value)


def _display_group_label(col, value, label_map=None):
    if not col or col == "None":
        return str(value)
    return _display_value(col, value, label_map)


def _display_model_term(term, original_covariates=None, label_map=None, data_raw=None):
    """Make Cox/AFT coefficient terms easier to read without changing the model."""
    original_covariates = original_covariates or []
    label_map = label_map or {}
    term_str = str(term)

    # Binary numeric covariate fitted as one-unit change, e.g. sex 1->2.
    if term_str in original_covariates and term_str in label_map and data_raw is not None and term_str in data_raw.columns:
        vals = sorted(data_raw[term_str].dropna().unique().tolist(), key=lambda x: str(x))
        if len(vals) == 2:
            low, high = vals[0], vals[1]
            return f"{term_str}: {_display_value(term_str, high, label_map)} compared with {_display_value(term_str, low, label_map)}"

    # Dummy variables created from categorical text columns, e.g. treatment_Intervention.
    for col in original_covariates:
        prefix = str(col) + "_"
        if term_str.startswith(prefix):
            level = term_str[len(prefix):]
            return f"{col}: {_display_value(col, level, label_map)} compared with reference"

    return term_str



def _term_info(term, original_covariates=None, label_map=None, data_raw=None):
    """Return structured, user-facing term labels for Cox/AFT tables.

    For a binary numeric predictor like sex coded 1/2, the model coefficient is
    a one-unit change from the lower value to the higher value. We display this
    as: Variable=sex, Level/Change=Female, Reference=Male. This is clearer than
    writing "Female compared with Male" inside one cell.
    """
    original_covariates = original_covariates or []
    label_map = label_map or {}
    term_str = str(term)

    # Binary numeric covariate fitted as one-unit change, e.g. sex 1 -> 2.
    if term_str in original_covariates and term_str in label_map and data_raw is not None and term_str in data_raw.columns:
        vals = sorted(data_raw[term_str].dropna().unique().tolist(), key=lambda x: str(x))
        if len(vals) == 2:
            low, high = vals[0], vals[1]
            return {
                "variable": term_str,
                "level": _display_value(term_str, high, label_map),
                "reference": _display_value(term_str, low, label_map),
                "display": f"{term_str}: {_display_value(term_str, high, label_map)}",
                "is_comparison": True,
            }

    # Dummy variables created from categorical text columns, e.g. treatment_Intervention.
    for col in original_covariates:
        prefix = str(col) + "_"
        if term_str.startswith(prefix):
            level = term_str[len(prefix):]
            return {
                "variable": str(col),
                "level": _display_value(col, level, label_map),
                "reference": "Reference group",
                "display": f"{col}: {_display_value(col, level, label_map)}",
                "is_comparison": True,
            }

    return {
        "variable": term_str,
        "level": "One-unit increase",
        "reference": "",
        "display": term_str,
        "is_comparison": False,
    }


def _plain_group_phrase(level, reference, outcome_text, pct, direction, estimate_text):
    """Clear natural-language sentence for binary/categorical comparisons."""
    if reference:
        return (
            f"**Plain interpretation:** **{level}** group has a **{pct:.1f}% {direction}** {outcome_text} "
            f"than **{reference}** group ({estimate_text}), holding other selected covariates constant."
        )
    return (
        f"**Plain interpretation:** A one-unit increase/change in **{level}** is associated with "
        f"a **{pct:.1f}% {direction}** {outcome_text} ({estimate_text}), holding other selected covariates constant."
    )


def _time_grid(data, duration_col, n_points=5):
    vals = pd.to_numeric(data[duration_col], errors="coerce").dropna()
    if vals.empty:
        return []
    qs = np.linspace(0.2, 0.8, n_points)
    times = sorted(set([float(np.quantile(vals, q)) for q in qs]))
    return times


def _survival_at_times(kmf, times):
    rows = []
    for t in times:
        try:
            s = float(kmf.predict(t))
        except Exception:
            s = np.nan
        rows.append({"Time": round(t, 3), "Survival probability": round(s, 4) if np.isfinite(s) else np.nan})
    return rows


def _make_risk_table(data, duration_col, event_col, group_col=None, times=None, label_map=None):
    if times is None:
        times = _time_grid(data, duration_col)
    rows = []
    if group_col and group_col in data.columns:
        for grp in sorted(data[group_col].dropna().unique(), key=lambda x: str(x)):
            d = data[data[group_col] == grp]
            row = {"Group": _display_group_label(group_col, grp, label_map)}
            for t in times:
                row[f"At risk @ {round(t, 2)}"] = int((d[duration_col] >= t).sum())
            rows.append(row)
    else:
        row = {"Group": "Overall"}
        for t in times:
            row[f"At risk @ {round(t, 2)}"] = int((data[duration_col] >= t).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _one_hot_for_lifelines(data, covariate_cols, protected_cols=None):
    """One-hot encode categorical covariates while preserving duration/event/strata columns."""
    protected_cols = protected_cols or []
    base_cols = [c for c in protected_cols if c in data.columns]
    cov_df = data[covariate_cols].copy()
    for c in cov_df.columns:
        if pd.api.types.is_bool_dtype(cov_df[c]):
            cov_df[c] = cov_df[c].astype(int)
    cat_cols = [
        c for c in cov_df.columns
        if pd.api.types.is_object_dtype(cov_df[c]) or pd.api.types.is_categorical_dtype(cov_df[c])
    ]
    if cat_cols:
        cov_df = pd.get_dummies(cov_df, columns=cat_cols, drop_first=True, dtype=float)
    for c in cov_df.columns:
        cov_df[c] = pd.to_numeric(cov_df[c], errors="coerce")
    model_df = pd.concat([data[base_cols].reset_index(drop=True), cov_df.reset_index(drop=True)], axis=1)
    model_df = model_df.dropna().copy()
    return model_df, list(cov_df.columns)


def _tie_summary(data, duration_col, event_col):
    event_times = data.loc[data[event_col] == 1, duration_col]
    if event_times.empty:
        return pd.DataFrame(), 0, 0
    vc = event_times.value_counts()
    tied_sets = vc[vc > 1]
    n_tied_events = int(tied_sets.sum()) if not tied_sets.empty else 0
    pct_tied_events = n_tied_events / max(int((data[event_col] == 1).sum()), 1) * 100
    tie_df = tied_sets.reset_index()
    tie_df.columns = ["Event time", "Number of tied events"]
    return tie_df.sort_values("Number of tied events", ascending=False), n_tied_events, pct_tied_events


# ============================================================
# Kaplan-Meier + Nelson-Aalen
# ============================================================

def run_kaplan_meier(df, duration_col, event_col, group_col, plot_template, show_nelson_aalen=True, label_map=None):
    st.markdown("## Kaplan-Meier Survival Analysis")
    st.caption(
        "Kaplan-Meier estimates the survival function S(t): the probability of remaining event-free beyond time t. "
        "It handles right-censored observations without assuming a parametric survival distribution."
    )

    for issue in _check_survival_inputs(df, duration_col, event_col):
        _show_issue(issue.get("level", "warning"), issue.get("msg", ""), issue.get("fix", ""))

    data = _prep_survival_data(df, duration_col, event_col, [group_col] if group_col else [])
    if data.empty:
        st.error("No data remaining after removing missing/invalid survival values.")
        return

    n_total = len(data)
    n_events = int(data[event_col].sum())
    n_censored = n_total - n_events
    pct_events = round(n_events / n_total * 100, 1) if n_total else 0
    pct_censored = round(n_censored / n_total * 100, 1) if n_total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total observations", f"{n_total:,}")
    c2.metric("Events", f"{n_events:,}")
    c3.metric("Censored", f"{n_censored:,}")
    c4.metric("Event rate", f"{pct_events}%")

    if n_events < 10:
        st.warning("Few events detected. Kaplan-Meier estimates and confidence intervals may be unstable.")
    if pct_censored > 70:
        st.warning(
            "High censoring rate. Survival analysis assumes censoring is non-informative; "
            "consider whether people censored early may differ systematically from those followed longer."
        )

    # Survival curve
    st.markdown("### Kaplan-Meier Survival Curve")
    fig = go.Figure()
    km_objects = {}
    group_used = bool(group_col and group_col in data.columns and group_col != "None")

    if group_used:
        groups = sorted(data[group_col].dropna().unique(), key=lambda x: str(x))
        if len(groups) > 10:
            st.warning("Too many groups (>10). Showing overall curve only. Choose a grouping variable with fewer categories.")
            group_used = False
        else:
            colors = px.colors.qualitative.Set2
            for i, grp in enumerate(groups):
                d = data[data[group_col] == grp]
                kmf = KaplanMeierFitter()
                display_grp = _display_group_label(group_col, grp, label_map)
                kmf.fit(d[duration_col], event_observed=d[event_col], label=display_grp)
                km_objects[display_grp] = kmf
                t = kmf.survival_function_.index.values
                s = kmf.survival_function_.iloc[:, 0].values
                ci_lower = kmf.confidence_interval_.iloc[:, 0].values
                ci_upper = kmf.confidence_interval_.iloc[:, 1].values
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=np.concatenate([t, t[::-1]]),
                    y=np.concatenate([ci_upper, ci_lower[::-1]]),
                    fill="toself",
                    fillcolor=color.replace("rgb", "rgba").replace(")", ",0.15)"),
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=t, y=s, mode="lines", name=f"{display_grp} (n={len(d)})",
                    line=dict(color=color, width=2, shape="hv"),
                ))
    if not group_used:
        kmf = KaplanMeierFitter()
        kmf.fit(data[duration_col], event_observed=data[event_col], label="Overall")
        km_objects["Overall"] = kmf
        t = kmf.survival_function_.index.values
        s = kmf.survival_function_.iloc[:, 0].values
        ci_lower = kmf.confidence_interval_.iloc[:, 0].values
        ci_upper = kmf.confidence_interval_.iloc[:, 1].values
        fig.add_trace(go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([ci_upper, ci_lower[::-1]]),
            fill="toself", fillcolor="rgba(37,99,235,0.15)",
            line=dict(color="rgba(255,255,255,0)"), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=t, y=s, mode="lines", name="Overall",
            line=dict(color="#2563EB", width=2, shape="hv"),
        ))

    fig.update_layout(
        title="Kaplan-Meier Survival Curve",
        xaxis_title=f"Time ({duration_col})",
        yaxis_title="Survival probability",
        yaxis=dict(range=[0, 1.05]),
        template=plot_template,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Median survival and survival probabilities
    st.markdown("### Median Survival and Survival Probabilities")
    med_rows = []
    time_points = _time_grid(data, duration_col, n_points=4)
    surv_prob_rows = []
    for label, kmf in km_objects.items():
        med = kmf.median_survival_time_
        med_rows.append({
            "Group": label,
            "Median survival time": round(float(med), 3) if med is not None and np.isfinite(med) else "Not reached",
            "Interpretation": "Time when estimated survival falls to 50%." if med is not None and np.isfinite(med) else "Survival curve did not fall below 50% during follow-up.",
        })
        for r in _survival_at_times(kmf, time_points):
            r["Group"] = label
            surv_prob_rows.append(r)
    st.dataframe(pd.DataFrame(med_rows), use_container_width=True)
    if surv_prob_rows:
        st.dataframe(pd.DataFrame(surv_prob_rows)[["Group", "Time", "Survival probability"]], use_container_width=True)

    # Risk table
    st.markdown("### Number at Risk Table")
    st.caption("The number at risk shows how many participants are still under observation and event-free at selected times.")
    risk_tbl = _make_risk_table(data, duration_col, event_col, group_col if group_used else None, times=time_points, label_map=label_map)
    st.dataframe(risk_tbl, use_container_width=True)

    # Log-rank test
    if group_used:
        groups = sorted(data[group_col].dropna().unique(), key=lambda x: str(x))
        st.markdown("### Log-Rank Test")
        st.caption("Compares survival curves between groups. It does not adjust for covariates.")
        try:
            if len(groups) == 2:
                d0 = data[data[group_col] == groups[0]]
                d1 = data[data[group_col] == groups[1]]
                lr = logrank_test(
                    d0[duration_col], d1[duration_col],
                    event_observed_A=d0[event_col], event_observed_B=d1[event_col],
                )
                l1, l2, l3 = st.columns(3)
                l1.metric("Test statistic", round(float(lr.test_statistic), 4))
                l2.metric("p-value", _format_p(float(lr.p_value)))
                l3.metric("Conclusion", "Significant" if lr.p_value < 0.05 else "Not significant")
                if lr.p_value < 0.05:
                    st.success("The survival curves differ statistically between groups. Check the KM plot to see which group has higher survival.")
                else:
                    st.info("No statistically significant survival difference was detected between groups.")
            elif len(groups) > 2:
                mlr = multivariate_logrank_test(data[duration_col], data[group_col], event_observed=data[event_col])
                l1, l2 = st.columns(2)
                l1.metric("Test statistic", round(float(mlr.test_statistic), 4))
                l2.metric("p-value", _format_p(float(mlr.p_value)))
        except Exception as e:
            st.warning("Log-rank test could not be completed: " + str(e))

    # Survival table for overall KM
    with st.expander("Kaplan-Meier survival table and download", expanded=False):
        kmf_tbl = KaplanMeierFitter()
        kmf_tbl.fit(data[duration_col], event_observed=data[event_col], label="Overall")
        tbl = kmf_tbl.survival_function_.copy()
        tbl.index = tbl.index.round(3)
        tbl.columns = ["Survival Probability"]
        tbl["CI Lower"] = kmf_tbl.confidence_interval_.iloc[:, 0].round(4)
        tbl["CI Upper"] = kmf_tbl.confidence_interval_.iloc[:, 1].round(4)
        tbl_out = tbl.reset_index().rename(columns={"timeline": "Time", "index": "Time"})
        st.dataframe(tbl_out, use_container_width=True)
        st.download_button(
            "Download Kaplan-Meier table (CSV)",
            data=tbl_out.to_csv(index=False).encode(),
            file_name="kaplan_meier_survival_table.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Nelson-Aalen cumulative hazard
    if show_nelson_aalen:
        st.markdown("### Nelson-Aalen Cumulative Hazard")
        st.caption(
            "Nelson-Aalen estimates cumulative hazard H(t). It complements Kaplan-Meier: "
            "S(t) decreases over time while cumulative hazard increases over time."
        )
        fig_h = go.Figure()
        colors = px.colors.qualitative.Set2
        na_tables = []
        try:
            if group_used:
                groups = sorted(data[group_col].dropna().unique(), key=lambda x: str(x))
                for i, grp in enumerate(groups):
                    d = data[data[group_col] == grp]
                    naf = NelsonAalenFitter()
                    display_grp = _display_group_label(group_col, grp, label_map)
                    naf.fit(d[duration_col], event_observed=d[event_col], label=display_grp)
                    h = naf.cumulative_hazard_
                    fig_h.add_trace(go.Scatter(
                        x=h.index, y=h.iloc[:, 0], mode="lines", name=display_grp,
                        line=dict(width=2, shape="hv", color=colors[i % len(colors)]),
                    ))
                    tmp = h.reset_index().rename(columns={"timeline": "Time", h.columns[0]: "Cumulative hazard"})
                    tmp["Group"] = display_grp
                    na_tables.append(tmp)
            else:
                naf = NelsonAalenFitter()
                naf.fit(data[duration_col], event_observed=data[event_col], label="Overall")
                h = naf.cumulative_hazard_
                fig_h.add_trace(go.Scatter(
                    x=h.index, y=h.iloc[:, 0], mode="lines", name="Overall",
                    line=dict(width=2, shape="hv", color="#2563EB"),
                ))
                tmp = h.reset_index().rename(columns={"timeline": "Time", h.columns[0]: "Cumulative hazard"})
                tmp["Group"] = "Overall"
                na_tables.append(tmp)
            fig_h.update_layout(
                title="Nelson-Aalen Cumulative Hazard",
                xaxis_title=f"Time ({duration_col})",
                yaxis_title="Cumulative hazard",
                template=plot_template,
            )
            st.plotly_chart(fig_h, use_container_width=True)
            if na_tables:
                na_df = pd.concat(na_tables, ignore_index=True)
                with st.expander("Nelson-Aalen cumulative hazard table", expanded=False):
                    st.dataframe(na_df.head(200), use_container_width=True)
                    st.download_button(
                        "Download Nelson-Aalen table (CSV)",
                        data=na_df.to_csv(index=False).encode(),
                        file_name="nelson_aalen_cumulative_hazard.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
        except Exception as e:
            st.warning("Nelson-Aalen estimate could not be completed: " + str(e))

    # Diagnostics / guidance
    st.markdown("### Diagnostics and Practical Warnings")
    issues = []
    if pct_events < 10:
        issues.append({"level": "error", "msg": f"Very low event rate ({pct_events}%).", "fix": "KM/log-rank results may be unstable; extend follow-up or collect more events."})
    elif pct_events < 30:
        issues.append({"level": "warning", "msg": f"Low event rate ({pct_events}%).", "fix": "Interpret confidence intervals and group comparisons cautiously."})
    if pct_censored > 70:
        issues.append({"level": "warning", "msg": f"High censoring rate ({pct_censored}%).", "fix": "Check whether censoring is likely non-informative; consider sensitivity analysis if not."})
    _diagnostic_summary(issues)


# ============================================================
# Cox Proportional Hazards
# ============================================================

def run_cox_ph(df, duration_col, event_col, covariate_cols, plot_template, strata_cols=None, label_map=None):
    st.markdown("## Cox Proportional Hazards Model")
    st.caption(
        "Cox PH estimates hazard ratios (HRs). HR > 1 means higher instantaneous event rate; HR < 1 means lower instantaneous event rate. "
        "The key assumption is proportional hazards: each HR is constant over time."
    )
    strata_cols = strata_cols or []
    strata_cols = [c for c in strata_cols if c and c != "None" and c in df.columns]

    extra_cols = list(dict.fromkeys(covariate_cols + strata_cols))
    data_raw = _prep_survival_data(df, duration_col, event_col, extra_cols)
    if data_raw.empty:
        st.error("No data remaining after removing missing/invalid survival values.")
        return

    # Remove rows with missing predictors/strata, then encode covariates
    data_raw = data_raw.dropna(subset=extra_cols).copy()
    protected = [duration_col, event_col] + strata_cols
    model_df, encoded_covs = _one_hot_for_lifelines(data_raw, covariate_cols, protected_cols=protected)

    if not encoded_covs:
        st.error("No usable covariates after encoding. Select at least one predictor.")
        return

    n_total = len(model_df)
    n_events = int(model_df[event_col].sum())
    epv = n_events / max(len(encoded_covs), 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{n_total:,}")
    c2.metric("Events", f"{n_events:,}")
    c3.metric("Event rate", f"{round(n_events/max(n_total,1)*100,1)}%")
    c4.metric("EPV", round(epv, 2), help="Events per estimated coefficient. Low EPV may indicate overfitting.")

    if strata_cols:
        st.info(
            "Stratified Cox is active. A separate baseline hazard is fitted for each stratum. "
            "The stratification variable itself will not have an HR estimate, because its effect is absorbed into the baseline hazard."
        )

    tie_df, n_tied_events, pct_tied = _tie_summary(model_df, duration_col, event_col)
    if n_tied_events > 0:
        with st.expander("Tied event times check", expanded=False):
            st.caption(
                "Tied event times occur when multiple events happen at the same observed time. "
                "Cox models commonly use approximations such as Efron/Breslow. A small number of ties is usually fine."
            )
            st.metric("Events occurring at tied times", f"{n_tied_events} ({pct_tied:.1f}%)")
            st.dataframe(tie_df.head(20), use_container_width=True)
            if pct_tied > 30:
                st.warning("Many events are tied. Interpret Cox estimates cautiously and consider whether time was recorded too coarsely.")

    try:
        cph = CoxPHFitter()
        cph.fit(
            model_df,
            duration_col=duration_col,
            event_col=event_col,
            strata=strata_cols if strata_cols else None,
            show_progress=False,
        )
    except Exception as e:
        st.error("Cox model failed to converge: " + str(e))
        st.info(
            "Common fixes: reduce predictors, remove highly sparse categories, check separation, "
            "standardize continuous variables, or use fewer dummy variables."
        )
        return

    st.success("Cox PH model fitted successfully.")

    try:
        c_idx = concordance_index(model_df[duration_col], -cph.predict_partial_hazard(model_df), model_df[event_col])
    except Exception:
        c_idx = np.nan
    m1, m2, m3 = st.columns(3)
    m1.metric("Concordance Index (C)", round(float(c_idx), 4) if np.isfinite(c_idx) else "N/A")
    m2.metric("Log-Likelihood", round(float(cph.log_likelihood_), 4))
    try:
        aic_val = cph.AIC_partial_
    except Exception:
        aic_val = np.nan
    m3.metric("Partial AIC", round(float(aic_val), 4) if np.isfinite(aic_val) else "N/A")

    # Coefficients
    st.markdown("### Coefficients — Hazard Ratios")
    summary = cph.summary.copy().reset_index()
    # lifelines column names vary slightly by version. After reset_index(),
    # the coefficient confidence interval columns may be named like
    # "coef lower 95%", "coef lower 95% ", or after normalization
    # "coef_lower_95%". Use a flexible lookup so the HR table works
    # across versions.
    summary.columns = [str(c).strip().replace(" ", "_") for c in summary.columns]

    def _get_summary_col(df, candidates):
        normalized = {str(c).strip().replace(" ", "_").lower(): c for c in df.columns}
        for cand in candidates:
            key = str(cand).strip().replace(" ", "_").lower()
            if key in normalized:
                return df[normalized[key]]
        raise KeyError("None of the expected columns were found: " + ", ".join(candidates))

    var_col = "covariate" if "covariate" in summary.columns else summary.columns[0]
    coef = _get_summary_col(summary, ["coef"])
    se_coef = _get_summary_col(summary, ["se(coef)", "se_coef"])
    coef_lower = _get_summary_col(summary, ["coef lower 95%", "coef_lower_95%", "coef lower 95% "])
    coef_upper = _get_summary_col(summary, ["coef upper 95%", "coef_upper_95%", "coef upper 95% "])
    p_value = _get_summary_col(summary, ["p", "p_value"])

    raw_terms = summary[var_col].astype(str)
    term_infos = [_term_info(v, covariate_cols, label_map, data_raw) for v in raw_terms]
    hr_vals = np.exp(coef)
    coef_tbl = pd.DataFrame({
        "Variable": [ti["variable"] for ti in term_infos],
        "Level / Change": [ti["level"] for ti in term_infos],
        "Reference": [ti["reference"] for ti in term_infos],
        "Coefficient": coef.round(4),
        "Std Error": se_coef.round(4),
        "Hazard Ratio (HR)": hr_vals.round(4),
        "HR CI Lower": np.exp(coef_lower).round(4),
        "HR CI Upper": np.exp(coef_upper).round(4),
        "p-value": p_value.round(5),
        "Interpretation": np.where(hr_vals > 1, "Higher hazard", "Lower hazard"),
    })
    # Hide empty reference column for purely continuous predictors.
    if (coef_tbl["Reference"].astype(str).str.strip() == "").all():
        coef_tbl = coef_tbl.drop(columns=["Reference"])
    st.dataframe(coef_tbl, use_container_width=True)

    # Compact interpretation sentence above the detailed plots.
    try:
        main = coef_tbl.iloc[0]
        hr = float(main["Hazard Ratio (HR)"])
        pval = float(main["p-value"])
        ci_lo = float(main["HR CI Lower"])
        ci_hi = float(main["HR CI Upper"])
        level = str(main["Level / Change"])
        reference = str(main.get("Reference", "")).strip()
        estimate = f"HR={hr:.3f}, 95% CI {ci_lo:.3f}–{ci_hi:.3f}, p={_format_p(pval)}"
        if hr < 1:
            pct = (1 - hr) * 100
            st.info(_plain_group_phrase(level, reference, "hazard of the event", pct, "lower", estimate))
        else:
            pct = (hr - 1) * 100
            st.info(_plain_group_phrase(level, reference, "hazard of the event", pct, "higher", estimate))
    except Exception:
        pass
    st.download_button(
        "Download Cox coefficients (CSV)",
        data=coef_tbl.to_csv(index=False).encode(),
        file_name="cox_coefficients.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Forest plot
    st.markdown("### Forest Plot — Hazard Ratios")
    coef_tbl_sorted = coef_tbl.sort_values("Hazard Ratio (HR)", ascending=True)
    forest_y = coef_tbl_sorted["Level / Change"] if "Level / Change" in coef_tbl_sorted.columns else coef_tbl_sorted["Variable"]
    fig_forest = go.Figure()
    fig_forest.add_trace(go.Scatter(
        x=coef_tbl_sorted["Hazard Ratio (HR)"],
        y=forest_y,
        mode="markers",
        marker=dict(size=10),
        error_x=dict(
            type="data",
            symmetric=False,
            array=(coef_tbl_sorted["HR CI Upper"] - coef_tbl_sorted["Hazard Ratio (HR)"]).values,
            arrayminus=(coef_tbl_sorted["Hazard Ratio (HR)"] - coef_tbl_sorted["HR CI Lower"]).values,
        ),
        name="HR (95% CI)",
    ))
    fig_forest.add_vline(x=1, line_dash="dash", annotation_text="HR = 1")
    fig_forest.update_layout(
        title="Hazard Ratio Forest Plot",
        xaxis_title="Hazard Ratio (log scale)",
        xaxis_type="log",
        template=plot_template,
        height=max(320, len(coef_tbl_sorted) * 38 + 120),
    )
    st.plotly_chart(fig_forest, use_container_width=True)

    # Baseline survival
    st.markdown("### Baseline Survival Curve")
    try:
        baseline = cph.baseline_survival_
        fig_base = go.Figure()
        fig_base.add_trace(go.Scatter(
            x=baseline.index, y=baseline.iloc[:, 0], mode="lines",
            line=dict(width=2, shape="hv"), name="Baseline survival",
        ))
        fig_base.update_layout(
            title="Baseline Survival Function",
            xaxis_title=f"Time ({duration_col})",
            yaxis_title="Survival probability",
            yaxis=dict(range=[0, 1.05]),
            template=plot_template,
        )
        st.plotly_chart(fig_base, use_container_width=True)
    except Exception as e:
        st.info("Baseline survival curve could not be displayed: " + str(e))

    # Diagnostics
    st.markdown("### Diagnostics and Model Warnings")
    issues = []
    if epv < 10:
        issues.append({
            "level": "warning" if epv >= 5 else "error",
            "msg": f"Events per variable is low (EPV = {epv:.1f}).",
            "fix": "Reduce the number of covariates/dummy categories or collect more events. Low EPV can produce unstable HR estimates.",
        })
    if np.isfinite(c_idx):
        if c_idx < 0.55:
            issues.append({"level": "warning", "msg": f"C-index is low ({c_idx:.3f}).", "fix": "The model has weak discrimination. Consider stronger predictors or interactions."})
        elif c_idx >= 0.70:
            st.success("C-index suggests good discrimination.")
        else:
            st.info("C-index suggests modest discrimination.")

    # PH test
    st.markdown("#### Proportional Hazards Assumption")
    st.caption(
        "This checks whether the hazard ratio stays roughly constant over time. p ≥ 0.05 means the PH assumption is acceptable; p < 0.05 suggests the effect may change over time."
    )
    ph_results = None
    try:
        ph_test = proportional_hazard_test(cph, model_df, time_transform="rank")
        ph_raw = ph_test.summary.copy().reset_index()

        # lifelines versions differ in how the proportional_hazard_test
        # summary is labelled. Some versions return columns like:
        # index, test_statistic, p, -log2(p). Keep only the columns
        # needed for display instead of forcing a fixed column count.
        first_col = ph_raw.columns[0]
        stat_col = next((c for c in ph_raw.columns if str(c).lower() in ["test_statistic", "test statistic", "chisq", "chi-square"]), None)
        p_col = next((c for c in ph_raw.columns if str(c).lower() in ["p", "p-value", "p_value"]), None)

        if stat_col is None:
            stat_candidates = [c for c in ph_raw.columns if c != first_col and pd.api.types.is_numeric_dtype(ph_raw[c])]
            stat_col = stat_candidates[0] if stat_candidates else None
        if p_col is None:
            p_candidates = [c for c in ph_raw.columns if str(c).lower().replace("_", "-") in ["p", "p-value"]]
            p_col = p_candidates[0] if p_candidates else None

        if stat_col is None or p_col is None:
            raise ValueError("Could not identify PH test statistic and p-value columns in lifelines output.")

        ph_infos = [_term_info(v, covariate_cols, label_map, data_raw) for v in ph_raw[first_col].astype(str)]
        ph_results = pd.DataFrame({
            "Variable": [ti["variable"] for ti in ph_infos],
            "Level / Change": [ti["level"] for ti in ph_infos],
            "Reference": [ti["reference"] for ti in ph_infos],
            "Test Statistic": pd.to_numeric(ph_raw[stat_col], errors="coerce"),
            "p-value": pd.to_numeric(ph_raw[p_col], errors="coerce"),
        })
        if (ph_results["Reference"].astype(str).str.strip() == "").all():
            ph_results = ph_results.drop(columns=["Reference"])
        ph_results["PH assumption"] = ph_results["p-value"].apply(lambda p: "Met" if pd.notna(p) and p >= 0.05 else "Violated")
        ph_results = ph_results.round(5)
        st.dataframe(ph_results, use_container_width=True)
        violated = ph_results[ph_results["p-value"] < 0.05]["Variable"].astype(str).tolist()
        if violated:
            issues.append({
                "level": "error",
                "msg": "PH assumption may be violated for: " + ", ".join(violated),
                "fix": "Use the stratification option for categorical variables, or consider a time-varying coefficient for continuous variables.",
            })
        else:
            st.success("No PH violation detected by the rank-based test.")
    except Exception as e:
        st.info("PH test could not be performed: " + str(e))

    # Martingale residuals
    st.markdown("#### Martingale Residuals: Linearity Check")
    st.caption(
        "For continuous predictors, Martingale residual plots help assess whether the predictor has a roughly linear effect on log hazard. "
        "A strong curve suggests transformation or spline terms."
    )
    try:
        martingale = model_df[event_col] - cph.predict_cumulative_hazard(model_df).iloc[-1]
        num_covariates = [c for c in encoded_covs if pd.api.types.is_numeric_dtype(model_df[c]) and model_df[c].nunique() > 5]
        if num_covariates:
            sel_cov = st.selectbox("Plot residuals against:", num_covariates, key="cox_mart_cov")
            fig_mart = go.Figure()
            fig_mart.add_trace(go.Scatter(
                x=model_df[sel_cov], y=martingale, mode="markers",
                marker=dict(size=5, opacity=0.55), name="Martingale residuals",
            ))
            try:
                lowess = __import__("statsmodels.api").api.nonparametric.lowess(martingale, model_df[sel_cov], frac=0.5)
                fig_mart.add_trace(go.Scatter(x=lowess[:, 0], y=lowess[:, 1], mode="lines", name="LOWESS trend"))
            except Exception:
                pass
            fig_mart.add_hline(y=0, line_dash="dash")
            fig_mart.update_layout(
                title=f"Martingale Residuals vs {sel_cov}",
                xaxis_title=sel_cov,
                yaxis_title="Martingale residual",
                template=plot_template,
            )
            st.plotly_chart(fig_mart, use_container_width=True)
        else:
            st.info("No continuous covariate available for Martingale residual plot.")
    except Exception as e:
        st.info("Martingale residuals could not be computed: " + str(e))

    # Log-log plot for categorical predictors
    st.markdown("#### Log-Log Plot: Visual PH Check")
    st.caption("Parallel log(-log(S(t))) curves support the PH assumption for categorical predictors.")
    try:
        cat_options = [c for c in covariate_cols if c in data_raw.columns and data_raw[c].nunique(dropna=True) <= 6]
        if cat_options:
            loglog_col = st.selectbox("Categorical variable for log-log plot:", cat_options, key="cox_loglog_col")
            fig_ll = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, grp in enumerate(sorted(data_raw[loglog_col].dropna().unique(), key=lambda x: str(x))):
                d = data_raw[data_raw[loglog_col] == grp]
                kmf_ll = KaplanMeierFitter()
                kmf_ll.fit(d[duration_col], event_observed=d[event_col])
                t = kmf_ll.survival_function_.index.values[1:]
                s = kmf_ll.survival_function_.iloc[1:, 0].values
                with np.errstate(divide="ignore", invalid="ignore"):
                    ll = np.log(-np.log(np.where(s > 0, s, np.nan)))
                fig_ll.add_trace(go.Scatter(
                    x=np.log(t + 1e-9), y=ll, mode="lines", name=_display_group_label(loglog_col, grp, label_map),
                    line=dict(width=2, color=colors[i % len(colors)]),
                ))
            fig_ll.update_layout(
                title="Log-Log Survival Plot",
                xaxis_title="log(Time)",
                yaxis_title="log(-log(S(t)))",
                template=plot_template,
            )
            st.plotly_chart(fig_ll, use_container_width=True)
        else:
            st.info("No categorical covariate with ≤6 categories is available for log-log plot.")
    except Exception as e:
        st.info("Log-log plot could not be generated: " + str(e))

    _diagnostic_summary(issues)

    # Model summary and PH fixes explanation
    with st.expander("Full Cox model summary", expanded=False):
        st.text(cph.summary.to_string())

    with st.expander("How to handle PH violations", expanded=False):
        st.markdown(
            """
**1) Stratified Cox**  
Use when a categorical variable violates PH. The model allows each stratum to have its own baseline hazard. You can still estimate HRs for the other covariates, but not for the stratification variable itself.

**2) Time-varying coefficients**  
Use when a predictor's effect clearly changes over follow-up time. For example, split follow-up at a clinically meaningful time point and allow the HR to differ before vs after that time.

**3) AFT / parametric survival model**  
Use when the PH assumption is not appropriate or when interpretation on the survival-time scale is preferable.
            """
        )


# ============================================================
# Parametric Survival: AFT models
# ============================================================

def run_parametric_aft(df, duration_col, event_col, covariate_cols, plot_template, dist_name="Weibull AFT", label_map=None):
    st.markdown("## Parametric Survival Model — AFT")
    st.caption(
        "Accelerated Failure Time (AFT) models estimate effects on the survival-time scale. "
        "A positive time-ratio interpretation means longer expected survival time; negative/less-than-one means shorter survival time."
    )

    data_raw = _prep_survival_data(df, duration_col, event_col, covariate_cols)
    data_raw = data_raw.dropna(subset=covariate_cols).copy()
    protected = [duration_col, event_col]
    model_df, encoded_covs = _one_hot_for_lifelines(data_raw, covariate_cols, protected_cols=protected)
    if not encoded_covs:
        st.error("No usable covariates after encoding.")
        return

    fitter_map = {
        "Weibull AFT": WeibullAFTFitter,
        "Log-normal AFT": LogNormalAFTFitter,
        "Log-logistic AFT": LogLogisticAFTFitter,
    }
    fitter_cls = fitter_map.get(dist_name, WeibullAFTFitter)

    try:
        aft = fitter_cls()
        aft.fit(model_df, duration_col=duration_col, event_col=event_col)
    except Exception as e:
        st.error("AFT model failed: " + str(e))
        st.info("Common fixes: reduce predictors, remove sparse categories, check positive/non-negative duration, or try another AFT distribution.")
        return

    st.success(dist_name + " fitted successfully.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Observations", f"{len(model_df):,}")
    c2.metric("Events", f"{int(model_df[event_col].sum()):,}")
    c3.metric("AIC", round(float(aft.AIC_), 3) if hasattr(aft, "AIC_") else "N/A")

    summ = aft.summary.copy().reset_index()

    # Simple AFT interpretation table first; the full lifelines table is kept below.
    st.markdown("### Time Ratio Interpretation")
    try:
        summ_clean = summ.copy()
        summ_clean.columns = [str(c).strip() for c in summ_clean.columns]
        param_col = "param" if "param" in summ_clean.columns else summ_clean.columns[0]
        cov_col = "covariate" if "covariate" in summ_clean.columns else summ_clean.columns[1]
        lambda_rows = summ_clean[
            (summ_clean[param_col].astype(str) == "lambda_")
            & (summ_clean[cov_col].astype(str).str.lower() != "intercept")
        ].copy()
        if not lambda_rows.empty:
            simple_rows = []
            for _, r in lambda_rows.iterrows():
                coef_val = float(r["coef"])
                tr = float(np.exp(coef_val))
                lo = float(r["exp(coef) lower 95%"])
                hi = float(r["exp(coef) upper 95%"])
                pval = float(r["p"])
                ti = _term_info(str(r[cov_col]), covariate_cols, label_map, data_raw)
                simple_rows.append({
                    "Variable": ti["variable"],
                    "Level / Change": ti["level"],
                    "Reference": ti["reference"],
                    "Time Ratio": round(tr, 4),
                    "95% CI Lower": round(lo, 4),
                    "95% CI Upper": round(hi, 4),
                    "p-value": round(pval, 5),
                    "Interpretation": "Longer survival time" if tr > 1 else "Shorter survival time",
                })
            simple_tbl = pd.DataFrame(simple_rows)
            if (simple_tbl["Reference"].astype(str).str.strip() == "").all():
                simple_tbl = simple_tbl.drop(columns=["Reference"])
            st.dataframe(simple_tbl, use_container_width=True)
            first = simple_tbl.iloc[0]
            tr = float(first["Time Ratio"])
            level = str(first["Level / Change"])
            reference = str(first.get("Reference", "")).strip()
            estimate = f"Time Ratio={tr:.3f}, p={_format_p(float(first['p-value']))}"
            if tr > 1:
                pct = (tr - 1) * 100
                st.info(_plain_group_phrase(level, reference, "survival time", pct, "longer", estimate))
            else:
                pct = (1 - tr) * 100
                st.info(_plain_group_phrase(level, reference, "survival time", pct, "shorter", estimate))
        else:
            st.info("No non-intercept AFT covariate terms available for a simplified time-ratio table.")
    except Exception as e:
        st.info("Simplified AFT interpretation table could not be created: " + str(e))

    with st.expander("Full AFT model coefficients", expanded=False):
        st.dataframe(summ, use_container_width=True)
        st.download_button(
            "Download AFT summary (CSV)",
            data=summ.to_csv(index=False).encode(),
            file_name="aft_model_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("### Predicted Survival Curves: Example Profiles")
    st.caption("Shows predicted survival for a few representative rows from the dataset. This is useful for explaining model behavior.")
    try:
        sample_rows = model_df[encoded_covs].drop_duplicates().head(3)
        times = np.linspace(float(model_df[duration_col].min()), float(model_df[duration_col].quantile(0.95)), 100)
        pred_surv = aft.predict_survival_function(sample_rows, times=times)
        fig = go.Figure()
        for i, col in enumerate(pred_surv.columns):
            fig.add_trace(go.Scatter(x=pred_surv.index, y=pred_surv[col], mode="lines", name=f"Profile {i+1}"))
        fig.update_layout(
            title=f"Predicted Survival Curves — {dist_name}",
            xaxis_title=f"Time ({duration_col})",
            yaxis_title="Predicted survival probability",
            yaxis=dict(range=[0, 1.05]),
            template=plot_template,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info("Predicted survival curves could not be displayed: " + str(e))

    with st.expander("How to choose an AFT distribution", expanded=False):
        st.markdown(
            """
**Weibull AFT**: flexible monotonic hazard; useful first parametric choice.  
**Log-normal AFT**: allows non-monotonic hazard; often useful when event risk rises then falls.  
**Log-logistic AFT**: also allows non-monotonic hazard and has interpretable survival-time effects.  

Compare AIC across candidate distributions. Lower AIC is preferred, but also inspect whether the fitted survival curves make clinical/analytical sense.
            """
        )


def run_parametric_model_comparison(df, duration_col, event_col, covariate_cols):
    st.markdown("### Parametric AFT Model Comparison")
    st.caption("Compares Weibull, log-normal, and log-logistic AFT models using AIC. Lower AIC is preferred.")
    data_raw = _prep_survival_data(df, duration_col, event_col, covariate_cols)
    data_raw = data_raw.dropna(subset=covariate_cols).copy()
    model_df, encoded_covs = _one_hot_for_lifelines(data_raw, covariate_cols, protected_cols=[duration_col, event_col])
    if not encoded_covs:
        return
    rows = []
    for name, cls in [
        ("Weibull AFT", WeibullAFTFitter),
        ("Log-normal AFT", LogNormalAFTFitter),
        ("Log-logistic AFT", LogLogisticAFTFitter),
    ]:
        try:
            fit = cls().fit(model_df, duration_col=duration_col, event_col=event_col)
            rows.append({"Model": name, "AIC": round(float(fit.AIC_), 3), "Log-likelihood": round(float(fit.log_likelihood_), 3), "Status": "Fitted"})
        except Exception as e:
            rows.append({"Model": name, "AIC": np.nan, "Log-likelihood": np.nan, "Status": "Failed: " + str(e)[:80]})
    comp = pd.DataFrame(rows).sort_values("AIC", na_position="last")
    st.dataframe(comp, use_container_width=True)


# ============================================================
# Main render function — called from app.py Tab
# ============================================================

def render_survival_tab(df, df_cleaned, plot_template):
    st.markdown("# 🫀 Survival Analysis")
    st.info(
        "Survival analysis models time until an event while accounting for censored observations. "
        "Use the warnings and recommendation boxes as guidance for choosing options and interpreting diagnostics."
    )

    dataset_choice = st.radio(
        "Dataset to use",
        ["Original data", "Cleaned data (from Data Cleaning tab)"],
        horizontal=True,
        key="surv_dataset",
    )
    mdf = df_cleaned.copy() if dataset_choice.startswith("Cleaned") else df.copy()
    all_cols = mdf.columns.tolist()
    numeric_cols = mdf.select_dtypes(include=np.number).columns.tolist()

    if len(all_cols) < 2:
        st.warning("Need at least duration and event/status columns.")
        return
    if not numeric_cols:
        st.warning("Need at least one numeric duration column.")
        return

    # Column setup and event coding
    st.markdown("### Column Setup")
    s1, s2, s3 = st.columns(3)
    with s1:
        duration_col = st.selectbox(
            "Duration column",
            numeric_cols,
            key="surv_duration",
            help="Time to event or censoring. Must be numeric and non-negative.",
        )
    with s2:
        event_options = [c for c in all_cols if c != duration_col]
        raw_event_col = st.selectbox(
            "Event/status column",
            event_options,
            key="surv_event_raw",
            help="This can be coded 0/1, 1/2, yes/no, dead/alive, etc.",
        )
    with s3:
        unique_vals = mdf[raw_event_col].dropna().unique().tolist()
        unique_vals_sorted = sorted(unique_vals, key=lambda x: str(x))
        default_idx = 0
        # Try to infer common event coding
        for common in [1, 2, "1", "2", True, "yes", "Yes", "dead", "Dead", "death", "Death", "event", "Event"]:
            if common in unique_vals_sorted:
                default_idx = unique_vals_sorted.index(common)
        event_value = st.selectbox(
            "Which value means EVENT occurred?",
            unique_vals_sorted,
            index=default_idx if unique_vals_sorted else 0,
            key="surv_event_value",
            help="Example: in hdat9600_lung, status=2 means death/event and status=1 means censored.",
        )

    if unique_vals_sorted:
        st.caption(
            "Event coding preview: values equal to **" + str(event_value) + "** are treated as **Event occurred**; "
            "all other non-missing values are treated as **Censored**. "
            "The app converts this internally for the survival model."
        )
    else:
        st.error("Selected event/status column has no non-missing values.")
        return

    mdf_model, event_col = _make_event_binary(mdf, raw_event_col, event_value)

    with st.expander("Event coding check", expanded=False):
        check_tbl = pd.DataFrame({
            "Raw event/status value": mdf[raw_event_col].dropna().astype(str).value_counts().index,
            "Count": mdf[raw_event_col].dropna().astype(str).value_counts().values,
        })
        check_tbl["Treated as"] = np.where(
            check_tbl["Raw event/status value"].astype(str) == str(event_value),
            "Event occurred",
            "Censored",
        )
        st.dataframe(check_tbl, use_container_width=True)
        event_rate = mdf_model[event_col].mean() * 100 if len(mdf_model) else 0
        st.metric("Derived event rate", f"{event_rate:.1f}%")
        if event_rate == 0 or event_rate == 100:
            st.warning("All rows were classified into only one outcome group. Check whether you selected the correct event value.")

    # Group column
    group_options = ["None"] + [c for c in all_cols if c not in (duration_col, raw_event_col)]
    group_col = st.selectbox(
        "Grouping column for Kaplan-Meier (optional)",
        group_options,
        key="surv_group",
    )

    # Model selection
    st.markdown("### Model Selection")
    model_type = st.radio(
        "Choose model",
        ["Kaplan-Meier / Nelson-Aalen", "Cox Proportional Hazards", "Parametric Survival (AFT)"],
        horizontal=True,
        key="surv_model_type",
    )

    covariate_cols = []
    strata_cols = []
    aft_dist = "Weibull AFT"
    if model_type in ["Cox Proportional Hazards", "Parametric Survival (AFT)"]:
        available_covs = [c for c in all_cols if c not in (duration_col, raw_event_col)]
        covariate_cols = st.multiselect(
            "Covariates / predictors",
            available_covs,
            default=available_covs[:min(5, len(available_covs))],
            key="surv_covariates",
        )

    if model_type == "Cox Proportional Hazards":
        with st.expander("Optional Cox settings", expanded=False):
            st.markdown(
                "**Stratification** is useful when a categorical predictor violates the PH assumption. "
                "It allows different baseline hazards by stratum, but no HR is estimated for the stratification variable."
            )
            strat_options = ["None"] + [c for c in all_cols if c not in (duration_col, raw_event_col) and mdf[c].nunique(dropna=True) <= 10]
            strat_selected = st.multiselect(
                "Stratification variable(s) optional",
                strat_options[1:],
                default=[],
                key="cox_strata_cols",
            )
            strata_cols = strat_selected

    if model_type == "Parametric Survival (AFT)":
        aft_dist = st.selectbox(
            "AFT distribution",
            ["Weibull AFT", "Log-normal AFT", "Log-logistic AFT"],
            key="surv_aft_dist",
            help="Compare distributions using AIC. Weibull is a good first parametric choice.",
        )
        compare_aft = st.checkbox("Also compare AIC across AFT distributions", value=True, key="aft_compare")
    else:
        compare_aft = False

    # Optional value labels are general: they work for sex, treatment, hospital, stage, etc.
    label_map = {}
    label_candidates = []
    if group_col and group_col != "None":
        label_candidates.append(group_col)
    label_candidates.extend(covariate_cols)
    label_candidates.extend(strata_cols)
    label_candidates = list(dict.fromkeys([c for c in label_candidates if c in all_cols and mdf[c].nunique(dropna=True) <= 10]))

    if label_candidates:
        with st.expander("Optional value labels for plots and interpretation", expanded=False):
            st.caption(
                "Use this only to rename coded values in the output. It does not change the analysis. "
                "Example: sex 1 = Male, 2 = Female; treatment 0 = Control, 1 = Intervention."
            )
            use_value_labels = st.checkbox("Use custom value labels", value=False, key="surv_use_value_labels")
            if use_value_labels:
                for col in label_candidates:
                    vals = sorted(mdf[col].dropna().unique().tolist(), key=lambda x: str(x))
                    st.markdown(f"**{col}**")
                    col_map = {}
                    ui_cols = st.columns(min(3, max(1, len(vals))))
                    for i, val in enumerate(vals):
                        default_label = str(val)
                        if str(col).lower() in ["sex", "gender"]:
                            if str(val) in ["1", "1.0"]:
                                default_label = "Male"
                            elif str(val) in ["2", "2.0"]:
                                default_label = "Female"
                            elif str(val).lower() in ["m", "male"]:
                                default_label = "Male"
                            elif str(val).lower() in ["f", "female"]:
                                default_label = "Female"
                        with ui_cols[i % len(ui_cols)]:
                            label = st.text_input(
                                f"{col} = {val}",
                                value=default_label,
                                key=f"surv_label_{col}_{str(val)}",
                            )
                        col_map[str(val)] = label
                    label_map[col] = col_map

    with st.expander("When to use each survival option", expanded=False):
        st.markdown(
            """
**Kaplan-Meier**: use for unadjusted survival curves and group comparison. It answers: *Do survival curves differ?*  
**Nelson-Aalen**: use to view cumulative hazard H(t). It complements KM and is useful when teaching hazard accumulation.  
**Cox PH**: use when you want adjusted hazard ratios while leaving baseline hazard unspecified. Always check PH assumption.  
**Stratified Cox**: use when a categorical variable violates PH and you still want HRs for the other variables.  
**Parametric AFT**: use when PH is questionable or when survival-time interpretation is preferred. Compare AIC across distributions.
            """
        )

    if st.button("▶ Run Survival Analysis", use_container_width=True, key="surv_run"):
        try:
            if model_type == "Kaplan-Meier / Nelson-Aalen":
                run_kaplan_meier(
                    mdf_model,
                    duration_col,
                    event_col,
                    None if group_col == "None" else group_col,
                    plot_template,
                    show_nelson_aalen=True,
                    label_map=label_map,
                )
            elif model_type == "Cox Proportional Hazards":
                if not covariate_cols:
                    st.error("Select at least one covariate for Cox PH.")
                else:
                    # Avoid using a variable both as covariate and strata.
                    covs = [c for c in covariate_cols if c not in strata_cols]
                    if not covs:
                        st.error("Select at least one covariate that is not used only for stratification.")
                    else:
                        run_cox_ph(mdf_model, duration_col, event_col, covs, plot_template, strata_cols=strata_cols, label_map=label_map)
            elif model_type == "Parametric Survival (AFT)":
                if not covariate_cols:
                    st.error("Select at least one covariate for AFT model.")
                else:
                    run_parametric_aft(mdf_model, duration_col, event_col, covariate_cols, plot_template, dist_name=aft_dist, label_map=label_map)
                    if compare_aft:
                        run_parametric_model_comparison(mdf_model, duration_col, event_col, covariate_cols)
        except Exception as e:
            st.error("Survival analysis error: " + str(e))
            with st.expander("Error details"):
                import traceback
                st.code(traceback.format_exc())

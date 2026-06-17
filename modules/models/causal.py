# ============================================================
# modules/models/causal.py
# Causal Inference: Multiple Matching Methods + IPW
# Step 1: Propensity Score (Logistic)
# Step 2: Matching (6 methods) + Common Support Trimming
# Step 3: Outcome Analysis (auto model selection) + ATE
# Extras: Rosenbaum Sensitivity, Subgroup Analysis, Placebo Test
# Full diagnostics + fix suggestions
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

import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv


# ============================================================
# Helpers
# ============================================================

def _show_issue(level, msg, fix=""):
    if level == "error":
        st.error("🔴 **" + msg + "**" + ("\n\n💡 *Fix:* " + fix if fix else ""))
    elif level == "warning":
        st.warning("🟡 **" + msg + "**" + ("\n\n💡 *Fix:* " + fix if fix else ""))
    else:
        st.info("🔵 " + msg)


def _diagnostic_summary(issues):
    if not issues:
        st.success("✅ All diagnostic checks passed — no major issues detected.")
    else:
        for i in issues:
            _show_issue(i["level"], i["msg"], i.get("fix", ""))


def _quote(col):
    safe = str(col).replace('"', '\\"')
    return 'Q("' + safe + '")'


def _smd(x1, x2):
    """Standardized Mean Difference between two groups."""
    mean1, mean2 = np.mean(x1), np.mean(x2)
    var1,  var2  = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_sd = np.sqrt((var1 + var2) / 2)
    if pooled_sd == 0:
        return 0.0
    return abs(mean1 - mean2) / pooled_sd



# ============================================================
# Matching helper functions (6 methods)
# ============================================================

def _match_psm(ps, treated_idx, control_idx, caliper, n_neighbors, with_replacement):
    """
    Nearest Neighbor matching on propensity score.
    Supports: 1:1, 1:k, with/without replacement.
    """
    ps_treated = ps[treated_idx].reshape(-1, 1)
    ps_control = ps[control_idx].reshape(-1, 1)

    k = max(n_neighbors, 1)
    search_k = min(len(control_idx), k * 5) if not with_replacement else k

    nn = NearestNeighbors(n_neighbors=search_k, metric="euclidean")
    nn.fit(ps_control)
    distances, indices = nn.kneighbors(ps_treated)

    matched_treated = []
    matched_control = []
    used_control = set()
    unmatched = 0

    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        valid = [(d, idx) for d, idx in zip(dists, idxs) if d <= caliper]

        if not with_replacement:
            valid = [(d, idx) for d, idx in valid if idx not in used_control]

        if not valid:
            unmatched += 1
            continue

        chosen = valid[:k]
        for d, idx in chosen:
            matched_treated.append(treated_idx[i])
            matched_control.append(control_idx[idx])
            if not with_replacement:
                used_control.add(idx)

    return matched_treated, matched_control, unmatched


def _match_mahalanobis(X_raw, treated_idx, control_idx, caliper_md, with_replacement):
    """Match on Mahalanobis distance over all covariates."""
    X = X_raw.values
    cov = np.cov(X, rowvar=False)
    try:
        inv_cov = inv(cov)
    except Exception:
        inv_cov = np.linalg.pinv(cov)

    X_treated = X[treated_idx]
    X_control = X[control_idx]

    matched_treated = []
    matched_control = []
    used_control = set()
    unmatched = 0

    for i, x_t in enumerate(X_treated):
        dists = []
        for j, x_c in enumerate(X_control):
            if not with_replacement and j in used_control:
                continue
            try:
                d = mahalanobis(x_t, x_c, inv_cov)
            except Exception:
                d = np.linalg.norm(x_t - x_c)
            dists.append((d, j))

        if not dists:
            unmatched += 1
            continue

        dists.sort(key=lambda x: x[0])
        best_d, best_j = dists[0]

        if best_d <= caliper_md:
            matched_treated.append(treated_idx[i])
            matched_control.append(control_idx[best_j])
            if not with_replacement:
                used_control.add(best_j)
        else:
            unmatched += 1

    return matched_treated, matched_control, unmatched


def _match_exact(data, treatment_col, exact_cols, treated_idx, control_idx):
    """Exact matching on selected categorical variables."""
    matched_treated = []
    matched_control = []
    unmatched = 0

    keys = data[exact_cols].astype(str).agg("|".join, axis=1)

    control_by_key = {}
    for idx in control_idx:
        k = keys.iloc[idx]
        control_by_key.setdefault(k, []).append(idx)

    used_control = set()

    for idx in treated_idx:
        k = keys.iloc[idx]
        candidates = [c for c in control_by_key.get(k, []) if c not in used_control]
        if candidates:
            chosen = candidates[0]
            matched_treated.append(idx)
            matched_control.append(chosen)
            used_control.add(chosen)
        else:
            unmatched += 1

    return matched_treated, matched_control, unmatched


def _match_mixed(data, treatment_col, exact_cols, ps,
                  treated_idx, control_idx, caliper, with_replacement):
    """Exact match on exact_cols, then PSM within each stratum."""
    keys = data[exact_cols].astype(str).agg("|".join, axis=1)

    matched_treated = []
    matched_control = []
    unmatched = 0

    treated_set = set(treated_idx)
    control_set = set(control_idx)

    unique_keys = keys.unique()
    for k in unique_keys:
        stratum_idx = np.where(keys.values == k)[0]
        s_treated = [i for i in stratum_idx if i in treated_set]
        s_control = [i for i in stratum_idx if i in control_set]

        if not s_treated:
            continue
        if not s_control:
            unmatched += len(s_treated)
            continue

        s_treated_arr = np.array(s_treated)
        s_control_arr = np.array(s_control)

        mt, mc, um = _match_psm(
            ps, s_treated_arr, s_control_arr, caliper,
            n_neighbors=1, with_replacement=with_replacement,
        )
        matched_treated.extend(mt)
        matched_control.extend(mc)
        unmatched += um

    return matched_treated, matched_control, unmatched


def compute_smart_caliper(ps, multiplier=0.2):
    """Standard rule: caliper = multiplier * SD(logit(propensity score))."""
    eps = 1e-6
    ps_clipped = np.clip(ps, eps, 1 - eps)
    logit_ps = np.log(ps_clipped / (1 - ps_clipped))
    sd_logit = np.std(logit_ps)
    caliper_logit = multiplier * sd_logit
    median_ps = np.median(ps_clipped)
    upper = 1 / (1 + np.exp(-(np.log(median_ps / (1 - median_ps)) + caliper_logit)))
    caliper_ps = abs(upper - median_ps)
    return round(float(caliper_ps), 4), round(float(sd_logit), 4)



# ============================================================
# Step 1: Propensity Score Estimation
# ============================================================

def step1_propensity_score(data, treatment_col, covariate_cols, plot_template):
    st.markdown("## Step 1: Propensity Score Estimation")
    st.caption(
        "Propensity score = probability of receiving treatment given covariates. "
        "Estimated via Logistic Regression."
    )

    issues = []

    # ── Prepare X and T ───────────────────────────────────────
    X_raw = data[covariate_cols].copy()

    # One-hot encode categoricals
    cat_cols = X_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if cat_cols:
        X_raw = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)

    # Fill any remaining NaN
    X_raw = X_raw.fillna(X_raw.median(numeric_only=True))

    T = data[treatment_col].astype(int).values
    X = X_raw.values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Fit Logistic ──────────────────────────────────────────
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_scaled, T)
    ps = lr.predict_proba(X_scaled)[:, 1]

    auc = roc_auc_score(T, ps)
    st.metric("AUC of Propensity Score Model", round(auc, 4),
              help="Good PS model: AUC between 0.6 and 0.9. Too high may indicate perfect separation.")

    if auc > 0.95:
        issues.append({
            "level": "warning",
            "msg": "AUC = " + str(round(auc, 4)) + " is very high - possible near-perfect separation.",
            "fix": "Check if any covariate perfectly predicts treatment. "
                   "Remove such covariates or use regularization.",
        })
    if auc < 0.55:
        issues.append({
            "level": "warning",
            "msg": "AUC = " + str(round(auc, 4)) + " is very low - PS model has poor discrimination.",
            "fix": "Add more relevant covariates to the propensity score model.",
        })

    # ── PS distribution by group ──────────────────────────────
    st.markdown("### Propensity Score Distribution")
    st.caption(
        "Good overlap (common support) is essential for valid causal inference. "
        "The two distributions should overlap substantially."
    )

    ps_df = pd.DataFrame({
        "Propensity Score": ps,
        "Group": ["Treated" if t == 1 else "Control" for t in T],
    })

    fig_ps = px.histogram(
        ps_df, x="Propensity Score", color="Group",
        barmode="overlay", opacity=0.6, nbins=40,
        color_discrete_map={"Treated": "#2563EB", "Control": "#DC2626"},
        title="Propensity Score Distribution by Group",
        template=plot_template,
    )
    fig_ps.update_layout(xaxis_title="Propensity Score", yaxis_title="Count")
    st.plotly_chart(fig_ps, use_container_width=True)

    # ── Common support check ──────────────────────────────────
    ps_treated = ps[T == 1]
    ps_control = ps[T == 0]

    overlap_min = max(ps_treated.min(), ps_control.min())
    overlap_max = min(ps_treated.max(), ps_control.max())
    overlap_pct = ((overlap_min <= ps) & (ps <= overlap_max)).mean() * 100

    o1, o2, o3 = st.columns(3)
    o1.metric("Common support min", round(overlap_min, 4))
    o2.metric("Common support max", round(overlap_max, 4))
    o3.metric("% in common support", round(overlap_pct, 1))

    if overlap_pct < 80:
        issues.append({
            "level": "error",
            "msg": "Poor common support - only " + str(round(overlap_pct, 1)) + "% of observations overlap.",
            "fix": "Restrict analysis to the region of common support. "
                   "Remove observations outside [" + str(round(overlap_min,3)) +
                   ", " + str(round(overlap_max,3)) + "].",
        })

    # ── ROC curve ─────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(T, ps)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        line=dict(color="#2563EB", width=2),
        name="PS Model (AUC=" + str(round(auc, 3)) + ")",
    ))
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="grey"))
    fig_roc.update_layout(
        title="ROC Curve - Propensity Score Model",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template=plot_template,
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # ── Coefficient table ──────────────────────────────────────
    st.markdown("### Propensity Score Model Coefficients")
    feat_names = list(X_raw.columns)
    coef_df = pd.DataFrame({
        "Covariate":    feat_names,
        "Coefficient":  [round(float(c), 4) for c in lr.coef_[0]],
        "Odds Ratio":   [round(float(np.exp(c)), 4) for c in lr.coef_[0]],
    }).sort_values("Odds Ratio", ascending=False)
    st.dataframe(coef_df, use_container_width=True)

    # Diagnostic summary
    st.markdown("### 🩺 Step 1 Diagnostics")
    _diagnostic_summary(issues)

    return ps, X_raw, issues



# ============================================================
# Common Support Trimming
# ============================================================

def apply_common_support_trimming(data, ps, treatment_col, trim_enabled=True):
    """
    Actually trims observations outside the common support region
    (instead of just warning about them).
    Returns trimmed data, trimmed ps, and a summary dict.
    """
    T = data[treatment_col].astype(int).values
    ps_treated = ps[T == 1]
    ps_control = ps[T == 0]

    overlap_min = max(ps_treated.min(), ps_control.min())
    overlap_max = min(ps_treated.max(), ps_control.max())

    in_support = (ps >= overlap_min) & (ps <= overlap_max)
    n_before = len(data)
    n_excluded = int((~in_support).sum())

    summary = {
        "overlap_min": round(float(overlap_min), 4),
        "overlap_max": round(float(overlap_max), 4),
        "n_before": n_before,
        "n_excluded": n_excluded,
        "pct_excluded": round(n_excluded / n_before * 100, 2) if n_before > 0 else 0,
    }

    if not trim_enabled:
        return data, ps, summary

    data_trimmed = data.iloc[in_support].reset_index(drop=True)
    ps_trimmed   = ps[in_support]

    return data_trimmed, ps_trimmed, summary


def render_common_support_section(data, ps, treatment_col, plot_template, trim_enabled):
    """Renders the common support plot + trimming summary + returns trimmed data."""
    st.markdown("### Common Support")
    st.caption(
        "Observations outside the overlapping propensity score region "
        "cannot be reliably matched and may bias the result if kept."
    )

    data_trimmed, ps_trimmed, summary = apply_common_support_trimming(
        data, ps, treatment_col, trim_enabled,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Common support range",
              "[" + str(summary["overlap_min"]) + ", " + str(summary["overlap_max"]) + "]")
    c2.metric("Observations excluded",
              str(summary["n_excluded"]) + " (" + str(summary["pct_excluded"]) + "%)")
    c3.metric("Trimming applied", "Yes" if trim_enabled else "No")

    T = data[treatment_col].astype(int).values
    ps_df = pd.DataFrame({
        "Propensity Score": ps,
        "Group": ["Treated" if t == 1 else "Control" for t in T],
    })
    fig = px.histogram(
        ps_df, x="Propensity Score", color="Group",
        barmode="overlay", opacity=0.6, nbins=40,
        color_discrete_map={"Treated": "#2563EB", "Control": "#DC2626"},
        title="Propensity Score with Common Support Region",
        template=plot_template,
    )
    fig.add_vline(x=summary["overlap_min"], line_dash="dash", line_color="green",
                 annotation_text="Support min")
    fig.add_vline(x=summary["overlap_max"], line_dash="dash", line_color="green",
                 annotation_text="Support max")
    st.plotly_chart(fig, use_container_width=True)

    if summary["pct_excluded"] > 20:
        st.warning(
            "More than 20% of observations fall outside common support. "
            "Your conclusions will apply only to the overlapping subpopulation, "
            "not to the full original sample."
        )
    elif trim_enabled and summary["n_excluded"] > 0:
        st.info(
            str(summary["n_excluded"]) + " observation(s) excluded for being "
            "outside the common support region."
        )

    return data_trimmed, ps_trimmed, summary


# ============================================================
# Step 2: Matching (dispatcher for all 6 methods)
# ============================================================

def step2_matching(data, treatment_col, covariate_cols, ps, X_raw,
                   matching_method, caliper, n_neighbors,
                   exact_cols, with_replacement, plot_template):
    st.markdown("## Step 2: Matching")

    method_descriptions = {
        "Nearest Neighbor (1:1)": "Each treated unit matched to the single closest control by propensity score.",
        "Nearest Neighbor (1:k)": "Each treated unit matched to k closest controls by propensity score.",
        "Nearest Neighbor with Replacement": "Same control unit may be reused across multiple treated units.",
        "Exact Matching": "Treated and control units must share identical values on the selected categorical variables.",
        "Exact + Propensity Score (Mixed)": "Exact match on selected categorical variables, then propensity score matching within each stratum.",
        "Mahalanobis Distance": "Matches on the multivariate distance across all covariates, not just the propensity score.",
    }
    st.caption(method_descriptions.get(matching_method, ""))

    issues = []
    T = data[treatment_col].astype(int).values

    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    n_treated = len(treated_idx)
    n_control = len(control_idx)

    st.markdown(
        "Treated: **" + str(n_treated) + "** | "
        "Control: **" + str(n_control) + "** | "
        "Ratio: **1:" + str(round(n_control / max(n_treated, 1), 1)) + "**"
    )

    if n_control < n_treated:
        issues.append({
            "level": "warning",
            "msg": "Fewer control units than treated units.",
            "fix": "Consider matching with replacement or using IPW instead.",
        })

    # ── Dispatch to the right matching function ───────────────
    if matching_method == "Nearest Neighbor (1:1)":
        matched_treated, matched_control, unmatched = _match_psm(
            ps, treated_idx, control_idx, caliper,
            n_neighbors=1, with_replacement=False,
        )

    elif matching_method == "Nearest Neighbor (1:k)":
        matched_treated, matched_control, unmatched = _match_psm(
            ps, treated_idx, control_idx, caliper,
            n_neighbors=n_neighbors, with_replacement=False,
        )

    elif matching_method == "Nearest Neighbor with Replacement":
        matched_treated, matched_control, unmatched = _match_psm(
            ps, treated_idx, control_idx, caliper,
            n_neighbors=1, with_replacement=True,
        )

    elif matching_method == "Exact Matching":
        if not exact_cols:
            st.error("Select at least one variable for Exact Matching.")
            return None, None
        matched_treated, matched_control, unmatched = _match_exact(
            data, treatment_col, exact_cols, treated_idx, control_idx,
        )

    elif matching_method == "Exact + Propensity Score (Mixed)":
        if not exact_cols:
            st.error("Select at least one variable for the Exact step of Mixed Matching.")
            return None, None
        matched_treated, matched_control, unmatched = _match_mixed(
            data, treatment_col, exact_cols, ps,
            treated_idx, control_idx, caliper, with_replacement,
        )

    elif matching_method == "Mahalanobis Distance":
        # caliper here is interpreted on Mahalanobis scale; use a generous default if too small
        md_caliper = max(caliper * 10, 1.0)
        matched_treated, matched_control, unmatched = _match_mahalanobis(
            X_raw, treated_idx, control_idx, md_caliper, with_replacement,
        )

    else:
        matched_treated, matched_control, unmatched = _match_psm(
            ps, treated_idx, control_idx, caliper,
            n_neighbors=1, with_replacement=False,
        )

    if not matched_treated:
        st.error(
            "No matches found with the current settings. "
            "Try increasing the caliper, choosing fewer exact-match variables, "
            "or switching matching method."
        )
        return None, None

    matched_idx = matched_treated + matched_control
    matched_data = data.iloc[matched_idx].copy()
    matched_ps   = ps[matched_idx]

    n_matched = len(matched_treated)
    match_rate = round(n_matched / n_treated * 100, 1)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matched pairs",     n_matched)
    m2.metric("Unmatched treated", unmatched)
    m3.metric("Match rate",        str(match_rate) + "%")
    m4.metric("Method",            matching_method.split(" (")[0])

    if match_rate < 70:
        issues.append({
            "level": "warning",
            "msg": "Match rate is " + str(match_rate) + "% - many treated units unmatched.",
            "fix": "Increase caliper size, allow replacement, or use IPW which uses all observations.",
        })

    # ── Matched PS distribution ───────────────────────────────
    st.markdown("### Matched Sample Propensity Score Distribution")
    T_matched = matched_data[treatment_col].astype(int).values
    ps_df_matched = pd.DataFrame({
        "Propensity Score": matched_ps,
        "Group": ["Treated" if t == 1 else "Control" for t in T_matched],
    })
    fig_mps = px.histogram(
        ps_df_matched, x="Propensity Score", color="Group",
        barmode="overlay", opacity=0.6, nbins=30,
        color_discrete_map={"Treated": "#2563EB", "Control": "#DC2626"},
        title="Propensity Score Distribution After Matching",
        template=plot_template,
    )
    st.plotly_chart(fig_mps, use_container_width=True)

    # ── Covariate balance — SMD before and after ──────────────
    st.markdown("### Covariate Balance")
    st.caption(
        "Standardized Mean Difference (SMD) < 0.1 indicates good balance. "
        "SMD should decrease substantially after matching."
    )

    num_covs = X_raw.select_dtypes(include=np.number).columns.tolist()
    balance_rows = []

    for col in num_covs:
        if col not in data.columns:
            continue

        before_t = data.loc[data[treatment_col] == 1, col].dropna()
        before_c = data.loc[data[treatment_col] == 0, col].dropna()
        smd_before = round(_smd(before_t.values, before_c.values), 4)

        after_t = matched_data.loc[matched_data[treatment_col] == 1, col].dropna()
        after_c = matched_data.loc[matched_data[treatment_col] == 0, col].dropna()
        smd_after = round(_smd(after_t.values, after_c.values), 4)

        balance_rows.append({
            "Covariate":    col,
            "SMD Before":   smd_before,
            "SMD After":    smd_after,
            "Improved":     "YES" if smd_after < smd_before else "NO",
            "Well Balanced": "YES" if smd_after < 0.10 else "NO",
        })

    balance_df = pd.DataFrame(balance_rows)

    if not balance_df.empty:
        st.dataframe(balance_df, use_container_width=True)

        st.markdown("### Love Plot")
        st.caption("Good matching moves SMD values closer to 0.")

        fig_love = go.Figure()
        fig_love.add_trace(go.Scatter(
            x=balance_df["SMD Before"],
            y=balance_df["Covariate"],
            mode="markers",
            marker=dict(size=10, color="#DC2626", symbol="circle"),
            name="Before Matching",
        ))
        fig_love.add_trace(go.Scatter(
            x=balance_df["SMD After"],
            y=balance_df["Covariate"],
            mode="markers",
            marker=dict(size=10, color="#16A34A", symbol="diamond"),
            name="After Matching",
        ))

        for _, row in balance_df.iterrows():
            fig_love.add_shape(
                type="line",
                x0=row["SMD Before"], y0=row["Covariate"],
                x1=row["SMD After"],  y1=row["Covariate"],
                line=dict(color="grey", width=1),
            )

        fig_love.add_vline(x=0.1,  line_dash="dash", line_color="orange",
                           annotation_text="SMD=0.1")
        fig_love.add_vline(x=0.0,  line_dash="dash", line_color="grey")
        fig_love.update_layout(
            title="Love Plot: Covariate Balance Before and After Matching",
            xaxis_title="Standardized Mean Difference (SMD)",
            template=plot_template,
            height=max(350, len(balance_df) * 35 + 100),
        )
        st.plotly_chart(fig_love, use_container_width=True)

        n_balanced = (balance_df["SMD After"] < 0.10).sum()
        n_total_cov = len(balance_df)
        st.metric(
            "Well-balanced covariates (SMD < 0.1)",
            str(n_balanced) + " / " + str(n_total_cov),
        )

        if n_balanced < n_total_cov * 0.7:
            issues.append({
                "level": "warning",
                "msg": "Balance is poor for some covariates after matching.",
                "fix": (
                    "Try: (1) Reduce caliper for stricter matching. "
                    "(2) Use IPW which often achieves better balance. "
                    "(3) Add remaining imbalanced covariates to outcome model (doubly robust)."
                ),
            })
        else:
            st.success("Good covariate balance achieved after matching.")

    st.markdown("### 🩺 Step 2 Diagnostics")
    _diagnostic_summary(issues)

    return matched_data, balance_df


# ============================================================
# Step 3: Outcome Analysis
# ============================================================

def step3_outcome(matched_data, treatment_col, outcome_col,
                  covariate_cols, outcome_type, plot_template):
    st.markdown("## Step 3: Outcome Analysis After Matching")
    st.caption(
        "Regression on the matched sample. "
        "The treatment coefficient = Average Treatment Effect on the Treated (ATT)."
    )

    issues = []
    data = matched_data.copy()

    n_treated = (data[treatment_col] == 1).sum()
    n_control = (data[treatment_col] == 0).sum()
    st.markdown(
        "Matched sample: **" + str(n_treated) + "** treated | "
        "**" + str(n_control) + "** control"
    )

    # ── Build formula ─────────────────────────────────────────
    # Treatment + adjusted covariates (doubly robust)
    cat_cols = data[covariate_cols].select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    parts = [_quote(treatment_col)]
    for col in covariate_cols:
        if col in cat_cols:
            parts.append("C(" + _quote(col) + ")")
        else:
            parts.append(_quote(col))

    formula = _quote(outcome_col) + " ~ " + " + ".join(parts)
    st.code(formula, language="text")

    # ── Fit outcome model ─────────────────────────────────────
    try:
        if outcome_type == "Continuous":
            result = smf.ols(formula=formula, data=data).fit()
            model_name = "Linear Regression"

        elif outcome_type == "Binary":
            unique_vals = data[outcome_col].dropna().unique()
            if not set(unique_vals).issubset({0, 1}):
                sorted_cls = sorted(unique_vals)
                data[outcome_col] = data[outcome_col].map(
                    {sorted_cls[0]: 0, sorted_cls[1]: 1}
                )
            result = smf.logit(formula=formula, data=data).fit(disp=False)
            model_name = "Logistic Regression"

        elif outcome_type == "Count":
            result = smf.glm(
                formula=formula, data=data,
                family=sm.families.Poisson()
            ).fit()
            model_name = "Poisson Regression"

        elif outcome_type == "Count (overdispersed)":
            result = smf.glm(
                formula=formula, data=data,
                family=sm.families.NegativeBinomial()
            ).fit()
            model_name = "Negative Binomial Regression"

        else:
            result = smf.ols(formula=formula, data=data).fit()
            model_name = "Linear Regression"

    except Exception as e:
        st.error("Outcome model failed: " + str(e))
        return

    st.success("Outcome model fitted: " + model_name)

    # ── ATT — main result ─────────────────────────────────────
    st.markdown("### Main Result: Average Treatment Effect on the Treated (ATT)")

    try:
        treat_param = result.params[_quote(treatment_col)]
        treat_pval  = result.pvalues[_quote(treatment_col)]
        treat_ci    = result.conf_int().loc[_quote(treatment_col)]
        treat_ci_lo = float(treat_ci[0])
        treat_ci_hi = float(treat_ci[1])
    except KeyError:
        # try without quotes
        try:
            treat_param = result.params[treatment_col]
            treat_pval  = result.pvalues[treatment_col]
            treat_ci    = result.conf_int().loc[treatment_col]
            treat_ci_lo = float(treat_ci[0])
            treat_ci_hi = float(treat_ci[1])
        except Exception:
            st.warning("Could not extract treatment coefficient directly.")
            treat_param = list(result.params)[1]
            treat_pval  = list(result.pvalues)[1]
            treat_ci_lo = float(list(result.conf_int()[0])[1])
            treat_ci_hi = float(list(result.conf_int()[1])[1])

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("ATT Estimate",  round(float(treat_param), 4))
    a2.metric("p-value",       round(float(treat_pval), 5))
    a3.metric("95% CI Lower",  round(treat_ci_lo, 4))
    a4.metric("95% CI Upper",  round(treat_ci_hi, 4))

    if outcome_type == "Binary":
        or_val = round(float(np.exp(treat_param)), 4)
        or_lo  = round(float(np.exp(treat_ci_lo)), 4)
        or_hi  = round(float(np.exp(treat_ci_hi)), 4)
        st.markdown(
            "**Odds Ratio: " + str(or_val) +
            "  (95% CI: " + str(or_lo) + " - " + str(or_hi) + ")**"
        )

    if outcome_type in ("Count", "Count (overdispersed)"):
        irr_val = round(float(np.exp(treat_param)), 4)
        irr_lo  = round(float(np.exp(treat_ci_lo)), 4)
        irr_hi  = round(float(np.exp(treat_ci_hi)), 4)
        st.markdown(
            "**IRR: " + str(irr_val) +
            "  (95% CI: " + str(irr_lo) + " - " + str(irr_hi) + ")**"
        )

    # Significance interpretation
    if float(treat_pval) < 0.05:
        direction = "higher" if float(treat_param) > 0 else "lower"
        st.success(
            "Treatment effect is statistically significant (p=" +
            str(round(float(treat_pval), 4)) + "). "
            "The treated group has significantly " + direction +
            " outcome compared to matched controls."
        )
    else:
        st.info(
            "Treatment effect is not statistically significant (p=" +
            str(round(float(treat_pval), 4)) + "). "
            "Cannot conclude a causal effect."
        )

    # ── Full coefficients table ───────────────────────────────
    st.markdown("### Full Coefficients Table")
    params_list = list(result.params)
    pvals_list  = list(result.pvalues)
    bse_list    = list(result.bse)
    conf_int    = result.conf_int()
    ci_lo_list  = list(conf_int[0])
    ci_hi_list  = list(conf_int[1])
    names_list  = list(result.params.index)

    coef_tbl = pd.DataFrame({
        "Term":        names_list,
        "Coefficient": [round(float(x), 4) for x in params_list],
        "Std Error":   [round(float(x), 4) for x in bse_list],
        "p-value":     [round(float(x), 5) for x in pvals_list],
        "CI Lower":    [round(float(x), 4) for x in ci_lo_list],
        "CI Upper":    [round(float(x), 4) for x in ci_hi_list],
        "Significant": ["YES" if float(p) < 0.05 else "NO" for p in pvals_list],
    })
    st.dataframe(coef_tbl, use_container_width=True)

    st.download_button(
        "📥 Download outcome model results (CSV)",
        data=coef_tbl.to_csv(index=False).encode(),
        file_name="causal_outcome_model.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Outcome distribution plot ──────────────────────────────
    st.markdown("### Outcome Distribution by Group")
    if outcome_type == "Continuous":
        fig_out = px.box(
            matched_data, x=treatment_col, y=outcome_col,
            color=treatment_col,
            title="Outcome Distribution: Treated vs Matched Controls",
            template=plot_template,
            labels={treatment_col: "Group (1=Treated, 0=Control)"},
        )
        st.plotly_chart(fig_out, use_container_width=True)

    elif outcome_type == "Binary":
        rate_df = matched_data.groupby(treatment_col)[outcome_col].mean().reset_index()
        rate_df.columns = ["Group", "Outcome Rate"]
        rate_df["Group"] = rate_df["Group"].map({1: "Treated", 0: "Control"})
        fig_out = px.bar(
            rate_df, x="Group", y="Outcome Rate",
            text="Outcome Rate",
            title="Outcome Rate by Group",
            template=plot_template,
            color="Group",
            color_discrete_map={"Treated": "#2563EB", "Control": "#DC2626"},
        )
        st.plotly_chart(fig_out, use_container_width=True)

    # ── Diagnostics ───────────────────────────────────────────
    if outcome_type == "Continuous":
        from statsmodels.stats.stattools import durbin_watson
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        resid = result.resid
        dw = durbin_watson(resid)
        st.metric("Durbin-Watson", round(dw, 4),
                  help="Ideal near 2.0")
        if dw < 1.5 or dw > 2.5:
            issues.append({
                "level": "warning",
                "msg": "Possible autocorrelation in residuals (DW=" + str(round(dw,3)) + ").",
                "fix": "Use robust standard errors or cluster standard errors by matched pair.",
            })

    if outcome_type == "Count":
        disp = result.pearson_chi2 / result.df_resid
        st.metric("Pearson chi2/df", round(disp, 4))
        if disp > 1.5:
            issues.append({
                "level": "error",
                "msg": "Overdispersion detected (chi2/df=" + str(round(disp,3)) + ").",
                "fix": "Switch outcome type to 'Count (overdispersed)' to use Negative Binomial.",
            })

    st.markdown("### 🩺 Step 3 Diagnostics")
    _diagnostic_summary(issues)

    with st.expander("Full Model Summary"):
        st.text(result.summary().as_text())

    return {
        "att_value": float(treat_param),
        "att_pval": float(treat_pval),
        "att_ci_lo": treat_ci_lo,
        "att_ci_hi": treat_ci_hi,
        "model_name": model_name,
    }



# ============================================================
# ATE estimation + ATT vs ATE comparison
# ============================================================

def estimate_ate(data_full, ps, treatment_col, outcome_col, outcome_type):
    """
    Estimate ATE using IPW on the full (unmatched) sample.
    ATE weight: treated = 1/ps, control = 1/(1-ps)
    """
    T = data_full[treatment_col].astype(int).values
    eps = 1e-6
    ps_clipped = np.clip(ps, eps, 1 - eps)

    weights_ate = np.where(T == 1, 1.0 / ps_clipped, 1.0 / (1 - ps_clipped))
    w_cap = np.percentile(weights_ate, 99)
    weights_ate_trimmed = np.clip(weights_ate, 0, w_cap)

    data = data_full.copy()

    try:
        if outcome_type == "Continuous":
            result = sm.WLS(
                data[outcome_col], sm.add_constant(data[treatment_col]),
                weights=weights_ate_trimmed,
            ).fit()
        else:
            result = sm.WLS(
                data[outcome_col].astype(float),
                sm.add_constant(data[treatment_col].astype(float)),
                weights=weights_ate_trimmed,
            ).fit()

        ate = float(result.params.iloc[1])
        ate_pval = float(result.pvalues.iloc[1])
        ate_ci = result.conf_int().iloc[1]
        ate_ci_lo, ate_ci_hi = float(ate_ci[0]), float(ate_ci[1])

        return {
            "ATE": round(ate, 4),
            "p-value": round(ate_pval, 5),
            "CI Lower": round(ate_ci_lo, 4),
            "CI Upper": round(ate_ci_hi, 4),
        }
    except Exception as e:
        return None


def render_att_vs_ate(att_value, att_pval, att_ci_lo, att_ci_hi,
                       data_full, ps, treatment_col, outcome_col, outcome_type):
    st.markdown("### ATT vs ATE")
    st.caption(
        "ATT = effect on those who actually received treatment. "
        "ATE = expected effect if treatment were applied to the entire population."
    )

    ate_result = estimate_ate(data_full, ps, treatment_col, outcome_col, outcome_type)

    if ate_result is None:
        st.info("ATE could not be estimated for this outcome type.")
        return

    compare_df = pd.DataFrame({
        "Estimand": ["ATT (matched sample)", "ATE (full sample, IPW)"],
        "Estimate": [round(att_value, 4), ate_result["ATE"]],
        "p-value":  [round(att_pval, 5), ate_result["p-value"]],
        "CI Lower": [round(att_ci_lo, 4), ate_result["CI Lower"]],
        "CI Upper": [round(att_ci_hi, 4), ate_result["CI Upper"]],
    })
    st.dataframe(compare_df, use_container_width=True)

    if abs(att_value - ate_result["ATE"]) > 0.2 * max(abs(att_value), 1e-6):
        direction = "larger" if abs(att_value) > abs(ate_result["ATE"]) else "smaller"
        st.info(
            "ATT is notably " + direction + " than ATE. "
            "This suggests treatment effects differ between those who self-selected "
            "into treatment and the broader population."
        )
    else:
        st.success("ATT and ATE are similar, suggesting a fairly consistent treatment effect across the population.")


# ============================================================
# Subgroup / Heterogeneous Treatment Effects
# ============================================================

def render_subgroup_analysis(matched_data, treatment_col, outcome_col,
                              outcome_type, all_cols, plot_template):
    st.markdown("### Subgroup Analysis — Heterogeneous Treatment Effects")
    st.caption(
        "Checks whether the treatment effect differs across subgroups "
        "(e.g., age groups, gender, severity level)."
    )

    candidate_cols = [
        c for c in all_cols
        if c in matched_data.columns
        and c not in (treatment_col, outcome_col)
        and matched_data[c].nunique() <= 10
    ]

    if not candidate_cols:
        st.info("No suitable categorical/low-cardinality variable found for subgroup analysis.")
        return

    subgroup_col = st.selectbox(
        "Subgroup variable",
        candidate_cols,
        key="ci_subgroup_col",
    )

    results = []
    for grp in sorted(matched_data[subgroup_col].dropna().unique()):
        sub = matched_data[matched_data[subgroup_col] == grp]
        n_t = (sub[treatment_col] == 1).sum()
        n_c = (sub[treatment_col] == 0).sum()

        if n_t < 3 or n_c < 3:
            results.append({
                "Subgroup": str(grp), "ATT": None, "p-value": None,
                "N": len(sub), "Note": "Too few observations",
            })
            continue

        try:
            if outcome_type == "Continuous":
                m = sm.OLS(
                    sub[outcome_col],
                    sm.add_constant(sub[treatment_col]),
                ).fit()
            else:
                m = sm.OLS(
                    sub[outcome_col].astype(float),
                    sm.add_constant(sub[treatment_col].astype(float)),
                ).fit()
            att = float(m.params.iloc[1])
            pval = float(m.pvalues.iloc[1])
            results.append({
                "Subgroup": str(grp), "ATT": round(att, 4),
                "p-value": round(pval, 5), "N": len(sub), "Note": "",
            })
        except Exception:
            results.append({
                "Subgroup": str(grp), "ATT": None, "p-value": None,
                "N": len(sub), "Note": "Model failed",
            })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)

    valid_results = results_df.dropna(subset=["ATT"])
    if not valid_results.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=valid_results["ATT"], y=valid_results["Subgroup"],
            mode="markers",
            marker=dict(
                size=12,
                color=["#16A34A" if p < 0.05 else "#94A3B8" for p in valid_results["p-value"]],
            ),
            name="ATT by subgroup",
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Treatment Effect by Subgroup (" + subgroup_col + ")",
            xaxis_title="ATT",
            template=plot_template,
            height=max(300, len(valid_results) * 50 + 100),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Green = statistically significant (p<0.05). Grey = not significant.")


# ============================================================
# Placebo Test
# ============================================================

def render_placebo_test(matched_data, treatment_col, all_cols,
                        outcome_col, plot_template):
    st.markdown("### Placebo Test")
    st.caption(
        "Runs the same analysis on an outcome that should NOT be affected by treatment "
        "(e.g., a pre-treatment measurement). A significant effect here would suggest "
        "hidden bias in the matching method."
    )

    candidate_outcomes = [
        c for c in all_cols
        if c in matched_data.columns
        and c not in (treatment_col, outcome_col)
        and pd.api.types.is_numeric_dtype(matched_data[c])
    ]

    if not candidate_outcomes:
        st.info("No suitable numeric variable available to use as a placebo outcome.")
        return

    placebo_col = st.selectbox(
        "Placebo outcome (should be unaffected by treatment)",
        candidate_outcomes,
        key="ci_placebo_col",
    )

    if st.button("Run Placebo Test", key="ci_placebo_run"):
        try:
            m = sm.OLS(
                matched_data[placebo_col].astype(float),
                sm.add_constant(matched_data[treatment_col].astype(float)),
            ).fit()
            att = float(m.params.iloc[1])
            pval = float(m.pvalues.iloc[1])

            c1, c2 = st.columns(2)
            c1.metric("Placebo effect estimate", round(att, 4))
            c2.metric("p-value", round(pval, 5))

            if pval >= 0.05:
                st.success(
                    "No significant effect on the placebo outcome (p=" + str(round(pval,4)) +
                    "). This supports the validity of the matching approach."
                )
            else:
                st.error(
                    "Significant effect detected on a variable that should be unaffected "
                    "(p=" + str(round(pval,4)) + "). This is a warning sign of hidden bias "
                    "in the matching — interpret the main result with caution."
                )
        except Exception as e:
            st.warning("Placebo test could not be run: " + str(e))


# ============================================================
# Rosenbaum Bounds Sensitivity Analysis
# ============================================================

def rosenbaum_bounds(matched_data, treatment_col, outcome_col, outcome_type,
                     gammas=None):
    """
    Rosenbaum sensitivity analysis for matched-pairs data.
    Approximation using Wilcoxon signed-rank style bounds for continuous outcomes,
    and McNemar-style bounds for binary outcomes on matched pairs.

    This requires the matched data to be organized in pairs (1 treated : 1 control).
    If matching produced 1:k or many-to-one matches, we use the first control
    per treated unit to form pairs for this analysis (clearly noted to the user).
    """
    if gammas is None:
        gammas = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    data = matched_data.copy()
    treated = data[data[treatment_col] == 1].reset_index(drop=True)
    control = data[data[treatment_col] == 0].reset_index(drop=True)

    n_pairs = min(len(treated), len(control))
    if n_pairs < 5:
        return None, "Not enough matched pairs (need at least 5) for sensitivity analysis."

    treated = treated.iloc[:n_pairs]
    control = control.iloc[:n_pairs]

    if outcome_type == "Continuous":
        diffs = treated[outcome_col].values - control[outcome_col].values
        # Wilcoxon signed-rank based bounds (approximate, large-sample normal approx)
        nonzero_diffs = diffs[diffs != 0]
        if len(nonzero_diffs) < 5:
            return None, "Not enough non-zero differences for sensitivity analysis."

        ranks = stats.rankdata(np.abs(nonzero_diffs))
        signs = np.sign(nonzero_diffs)
        W = np.sum(ranks[signs > 0])
        n = len(nonzero_diffs)

        results = []
        for gamma in gammas:
            p_plus = gamma / (1 + gamma)
            p_minus = 1 / (1 + gamma)

            mean_W_plus = p_plus * np.sum(ranks)
            var_W = p_plus * p_minus * np.sum(ranks ** 2)

            z_upper = (W - mean_W_plus) / np.sqrt(var_W) if var_W > 0 else 0
            p_upper = 1 - stats.norm.cdf(z_upper)

            mean_W_minus = p_minus * np.sum(ranks)
            z_lower = (W - mean_W_minus) / np.sqrt(var_W) if var_W > 0 else 0
            p_lower = 1 - stats.norm.cdf(z_lower)

            results.append({
                "Gamma": gamma,
                "p-value (Lower bound)": round(float(min(p_lower, p_upper)), 5),
                "p-value (Upper bound)": round(float(max(p_lower, p_upper)), 5),
            })

    else:
        # Binary outcome: McNemar-style sign test bounds
        diffs = treated[outcome_col].values.astype(float) - control[outcome_col].values.astype(float)
        n_plus = int((diffs > 0).sum())   # treated better
        n_minus = int((diffs < 0).sum())  # control better
        n_disc = n_plus + n_minus

        if n_disc < 5:
            return None, "Not enough discordant pairs for sensitivity analysis."

        results = []
        for gamma in gammas:
            p_plus = gamma / (1 + gamma)
            p_minus = 1 / (1 + gamma)

            p_upper = 1 - stats.binom.cdf(n_plus - 1, n_disc, p_plus)
            p_lower = 1 - stats.binom.cdf(n_plus - 1, n_disc, p_minus)

            results.append({
                "Gamma": gamma,
                "p-value (Lower bound)": round(float(min(p_lower, p_upper)), 5),
                "p-value (Upper bound)": round(float(max(p_lower, p_upper)), 5),
            })

    return pd.DataFrame(results), None


def render_rosenbaum_section(matched_data, treatment_col, outcome_col,
                             outcome_type, plot_template):
    st.markdown("### Sensitivity Analysis — Rosenbaum Bounds")
    st.caption(
        "Tests how robust the result is to a hidden (unmeasured) confounder. "
        "Gamma = the strength of hidden bias needed to overturn the conclusion. "
        "Higher Gamma at which p stays below 0.05 means a more robust result."
    )

    bounds_df, error_msg = rosenbaum_bounds(
        matched_data, treatment_col, outcome_col, outcome_type,
    )

    if error_msg:
        st.info(error_msg)
        return

    st.dataframe(bounds_df, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bounds_df["Gamma"], y=bounds_df["p-value (Upper bound)"],
        mode="lines+markers", name="Upper bound p-value",
        line=dict(color="#DC2626", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=bounds_df["Gamma"], y=bounds_df["p-value (Lower bound)"],
        mode="lines+markers", name="Lower bound p-value",
        line=dict(color="#2563EB", width=2),
    ))
    fig.add_hline(y=0.05, line_dash="dash", line_color="orange",
                 annotation_text="p = 0.05")
    fig.update_layout(
        title="Rosenbaum Sensitivity Bounds",
        xaxis_title="Gamma (hidden bias strength)",
        yaxis_title="p-value",
        template=plot_template,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Find the critical gamma where upper bound crosses 0.05
    crit_rows = bounds_df[bounds_df["p-value (Upper bound)"] >= 0.05]
    if not crit_rows.empty:
        crit_gamma = crit_rows["Gamma"].iloc[0]
        st.warning(
            "Result becomes non-significant at Gamma = " + str(crit_gamma) + ". "
            "A hidden confounder with this strength (or stronger) could explain away the effect."
        )
    else:
        max_gamma = bounds_df["Gamma"].max()
        st.success(
            "Result remains significant up to Gamma = " + str(max_gamma) +
            " (the highest tested value). This suggests a robust finding, "
            "though stronger hidden bias was not tested."
        )

    st.caption(
        "Note: this analysis approximates matched pairs from the first control "
        "matched to each treated unit when matching was 1:k."
    )


# ============================================================
# IPW - Inverse Probability Weighting
# ============================================================

def run_ipw(data, treatment_col, outcome_col,
            covariate_cols, ps, outcome_type, plot_template):
    st.markdown("## Alternative: Inverse Probability Weighting (IPW)")
    st.caption(
        "IPW uses all observations (no units discarded). "
        "Each observation is weighted by the inverse of its propensity score. "
        "ATT weights: treated=1, control=ps/(1-ps)."
    )

    issues = []
    T = data[treatment_col].astype(int).values

    # ── Compute ATT weights ───────────────────────────────────
    eps = 1e-6
    ps_clipped = np.clip(ps, eps, 1 - eps)

    weights = np.where(T == 1, 1.0, ps_clipped / (1 - ps_clipped))

    # Trim extreme weights (top 1%)
    w_cap = np.percentile(weights, 99)
    weights_trimmed = np.clip(weights, 0, w_cap)

    ess = (weights_trimmed.sum() ** 2) / (weights_trimmed ** 2).sum()

    w1, w2, w3 = st.columns(3)
    w1.metric("Max weight (trimmed)", round(float(w_cap), 3))
    w2.metric("Effective Sample Size", round(float(ess), 1))
    w3.metric("Original N", len(data))

    if ess < len(data) * 0.3:
        issues.append({
            "level": "warning",
            "msg": "Low effective sample size (ESS=" + str(round(float(ess),1)) + "). Weights are very unequal.",
            "fix": "Trim more extreme weights or use matching instead.",
        })

    # ── Weight distribution ───────────────────────────────────
    w_df = pd.DataFrame({
        "Weight": weights_trimmed,
        "Group": ["Treated" if t == 1 else "Control" for t in T],
    })
    fig_w = px.histogram(
        w_df, x="Weight", color="Group",
        barmode="overlay", opacity=0.6, nbins=30,
        color_discrete_map={"Treated": "#2563EB", "Control": "#DC2626"},
        title="IPW Weight Distribution",
        template=plot_template,
    )
    st.plotly_chart(fig_w, use_container_width=True)

    # ── Weighted outcome model ────────────────────────────────
    st.markdown("### Weighted Outcome Model")
    cat_cols = data[covariate_cols].select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    parts = [_quote(treatment_col)]
    for col in covariate_cols:
        if col in cat_cols:
            parts.append("C(" + _quote(col) + ")")
        else:
            parts.append(_quote(col))
    formula = _quote(outcome_col) + " ~ " + " + ".join(parts)

    try:
        if outcome_type == "Continuous":
            result = smf.wls(
                formula=formula, data=data, weights=weights_trimmed
            ).fit()
        elif outcome_type == "Binary":
            unique_vals = data[outcome_col].dropna().unique()
            if not set(unique_vals).issubset({0, 1}):
                sorted_cls = sorted(unique_vals)
                data[outcome_col] = data[outcome_col].map(
                    {sorted_cls[0]: 0, sorted_cls[1]: 1}
                )
            result = smf.glm(
                formula=formula, data=data,
                family=sm.families.Binomial(),
                freq_weights=weights_trimmed.astype(int),
            ).fit()
        else:
            result = smf.wls(
                formula=formula, data=data, weights=weights_trimmed
            ).fit()

        # ATT
        try:
            treat_param = float(result.params[_quote(treatment_col)])
            treat_pval  = float(result.pvalues[_quote(treatment_col)])
        except KeyError:
            treat_param = float(list(result.params)[1])
            treat_pval  = float(list(result.pvalues)[1])

        i1, i2 = st.columns(2)
        i1.metric("IPW ATT Estimate", round(treat_param, 4))
        i2.metric("p-value",          round(treat_pval, 5))

        if treat_pval < 0.05:
            st.success("IPW: Treatment effect significant (p=" + str(round(treat_pval,4)) + ").")
        else:
            st.info("IPW: Treatment effect not significant (p=" + str(round(treat_pval,4)) + ").")

        with st.expander("Full IPW Model Summary"):
            st.text(result.summary().as_text())

    except Exception as e:
        st.warning("IPW outcome model could not be fitted: " + str(e))

    st.markdown("### 🩺 IPW Diagnostics")
    _diagnostic_summary(issues)



# ============================================================
# HTML Report Generation
# ============================================================

def generate_causal_html_report(
    treatment_col, outcome_col, outcome_type, matching_method,
    n_treated_orig, n_control_orig,
    auc, overlap_pct,
    n_matched, match_rate, balance_df,
    att_value, att_pval, att_ci_lo, att_ci_hi,
    ate_result,
    rosenbaum_df,
):
    balance_html = (
        balance_df.to_html(index=False)
        if balance_df is not None and not balance_df.empty
        else "<p>No balance table available.</p>"
    )
    rosenbaum_html = (
        rosenbaum_df.to_html(index=False)
        if rosenbaum_df is not None and not rosenbaum_df.empty
        else "<p>Sensitivity analysis not available (insufficient matched pairs).</p>"
    )
    ate_html = (
        "<p>ATE: " + str(ate_result["ATE"]) + " (95% CI: " +
        str(ate_result["CI Lower"]) + " to " + str(ate_result["CI Upper"]) +
        ", p=" + str(ate_result["p-value"]) + ")</p>"
        if ate_result else "<p>ATE not available.</p>"
    )

    significance = "statistically significant" if att_pval < 0.05 else "not statistically significant"

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Causal Inference Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; background: #F8FAFC; color: #0F172A; }}
  .header {{ background: linear-gradient(135deg, #1E3A8A, #2563EB); color: white;
             padding: 30px; border-radius: 16px; margin-bottom: 30px; }}
  .header h1 {{ margin: 0; }}
  .card {{ background: white; padding: 20px; border-radius: 14px; margin-bottom: 25px;
           border: 1px solid #E2E8F0; }}
  h2 {{ color: #1E3A8A; border-bottom: 2px solid #E2E8F0; padding-bottom: 8px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 15px; font-size: 14px; }}
  th, td {{ border: 1px solid #CBD5E1; padding: 8px; text-align: left; }}
  th {{ background: #E0F2FE; }}
  .metric {{ display: inline-block; background: #F1F5F9; border-radius: 10px;
             padding: 12px 18px; margin: 6px; min-width: 140px; text-align: center; }}
  .metric-value {{ font-size: 22px; font-weight: bold; color: #2563EB; }}
  .metric-label {{ font-size: 12px; color: #64748B; }}
</style>
</head>
<body>
<div class="header">
  <h1>Causal Inference Report</h1>
  <p>Treatment: {treatment_col} &mdash; Outcome: {outcome_col} ({outcome_type})</p>
</div>

<div class="card">
  <h2>1. Executive Summary</h2>
  <p>The estimated Average Treatment Effect on the Treated (ATT) is
  <strong>{att_value}</strong> (95% CI: {att_ci_lo} to {att_ci_hi}, p={att_pval}).
  This result is <strong>{significance}</strong>.</p>
</div>

<div class="card">
  <h2>2. Method</h2>
  <p>Propensity scores were estimated using logistic regression. Matching method:
  <strong>{matching_method}</strong>.</p>
  <div class="metric"><div class="metric-value">{n_treated_orig}</div><div class="metric-label">Original Treated</div></div>
  <div class="metric"><div class="metric-value">{n_control_orig}</div><div class="metric-label">Original Control</div></div>
  <div class="metric"><div class="metric-value">{auc}</div><div class="metric-label">PS Model AUC</div></div>
  <div class="metric"><div class="metric-value">{overlap_pct}%</div><div class="metric-label">Common Support</div></div>
</div>

<div class="card">
  <h2>3. Matching Quality</h2>
  <div class="metric"><div class="metric-value">{n_matched}</div><div class="metric-label">Matched Pairs</div></div>
  <div class="metric"><div class="metric-value">{match_rate}%</div><div class="metric-label">Match Rate</div></div>
  {balance_html}
</div>

<div class="card">
  <h2>4. Treatment Effect</h2>
  <div class="metric"><div class="metric-value">{att_value}</div><div class="metric-label">ATT</div></div>
  <div class="metric"><div class="metric-value">{att_pval}</div><div class="metric-label">p-value</div></div>
  <p>95% Confidence Interval: [{att_ci_lo}, {att_ci_hi}]</p>
  {ate_html}
</div>

<div class="card">
  <h2>5. Sensitivity Analysis (Rosenbaum Bounds)</h2>
  <p>Tests robustness of the result to unmeasured confounding.</p>
  {rosenbaum_html}
</div>

<div class="card">
  <h2>6. Conclusion</h2>
  <p>Based on the matched-sample analysis, the treatment effect is {significance}.
  Users should review the covariate balance table and sensitivity analysis above
  before drawing causal conclusions, and confirm that the ignorability and
  positivity assumptions are plausible for this dataset.</p>
</div>

</body>
</html>
"""


# ============================================================
# Main render function
# ============================================================

def render_causal_tab(df, df_cleaned, plot_template):
    st.markdown("# Causal Inference")
    st.info(
        "Estimates the causal effect of a treatment or intervention "
        "by balancing observed confounders between treated and control groups."
    )

    dataset_choice = st.radio(
        "Dataset to use",
        ["Original data", "Cleaned data (from Data Cleaning tab)"],
        horizontal=True,
        key="ci_dataset",
    )
    mdf = df_cleaned.copy() if dataset_choice.startswith("Cleaned") else df.copy()

    all_cols     = mdf.columns.tolist()
    numeric_cols = mdf.select_dtypes(include=np.number).columns.tolist()
    cat_cols_all = mdf.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if len(all_cols) < 3:
        st.warning("Need at least 3 columns: treatment + outcome + covariates.")
        return

    # ── Column setup ──────────────────────────────────────────
    st.markdown("### Column Setup")
    ci1, ci2, ci3 = st.columns(3)

    with ci1:
        treatment_col = st.selectbox(
            "Treatment variable (1=treated, 0=control)",
            all_cols,
            key="ci_treatment",
            help="Must be binary: 1 = received treatment, 0 = control group.",
        )
    with ci2:
        outcome_col = st.selectbox(
            "Outcome variable (Y)",
            [c for c in all_cols if c != treatment_col],
            key="ci_outcome",
        )
    with ci3:
        outcome_type = st.selectbox(
            "Outcome type",
            ["Continuous", "Binary", "Count", "Count (overdispersed)"],
            key="ci_outcome_type",
        )

    # ── Immediate readiness checks ─────────────────────────────
    readiness_issues = []

    unique_treat_vals = mdf[treatment_col].dropna().unique()
    if len(unique_treat_vals) != 2:
        readiness_issues.append(
            "Treatment column '" + treatment_col + "' has " +
            str(len(unique_treat_vals)) + " unique values, but must have exactly 2."
        )

    if pd.api.types.is_numeric_dtype(mdf[outcome_col]):
        unique_outcome_vals = mdf[outcome_col].dropna().unique()
        if outcome_type == "Binary" and len(unique_outcome_vals) > 2:
            readiness_issues.append(
                "Outcome type is set to 'Binary' but '" + outcome_col +
                "' has " + str(len(unique_outcome_vals)) + " unique values."
            )
        if outcome_type in ("Count", "Count (overdispersed)"):
            if (mdf[outcome_col].dropna() < 0).any():
                readiness_issues.append(
                    "Outcome type is set to '" + outcome_type +
                    "' but '" + outcome_col + "' contains negative values."
                )
    else:
        if outcome_type != "Binary":
            readiness_issues.append(
                "'" + outcome_col + "' contains text/categories. "
                "Set Outcome type to 'Binary' or choose a numeric outcome column."
            )

    n_treated_check = int((mdf[treatment_col] == unique_treat_vals[0]).sum()) if len(unique_treat_vals) > 0 else 0
    if len(mdf) < 20:
        readiness_issues.append(
            "Dataset has only " + str(len(mdf)) + " rows. Causal analysis needs at least 20."
        )

    if readiness_issues:
        for issue in readiness_issues:
            st.warning(issue)
    else:
        st.success("Column setup looks valid. Configure covariates and matching below.")

    available_covs = [c for c in all_cols if c not in (treatment_col, outcome_col)]
    covariate_cols = st.multiselect(
        "Confounders / covariates for propensity score",
        available_covs,
        default=available_covs[:min(6, len(available_covs))],
        key="ci_covariates",
        help="Variables that affect both treatment assignment and outcome.",
    )

    # ── Matching method selection ─────────────────────────────
    st.markdown("### Matching Method")

    MATCHING_METHODS = [
        "Nearest Neighbor (1:1)",
        "Nearest Neighbor (1:k)",
        "Nearest Neighbor with Replacement",
        "Exact Matching",
        "Exact + Propensity Score (Mixed)",
        "Mahalanobis Distance",
    ]
    matching_method = st.selectbox(
        "Choose matching method",
        MATCHING_METHODS,
        key="ci_matching_method",
    )

    # ── Method-specific settings ──────────────────────────────
    exact_cols = []
    n_neighbors = 1
    with_replacement = False

    if matching_method == "Nearest Neighbor (1:k)":
        n_neighbors = st.slider(
            "k (number of control matches per treated unit)",
            2, 5, 2, key="ci_k_neighbors",
        )

    if matching_method == "Nearest Neighbor with Replacement":
        with_replacement = True

    if matching_method in ("Exact Matching", "Exact + Propensity Score (Mixed)"):
        exact_candidates = [
            c for c in covariate_cols
            if mdf[c].nunique() <= 15
        ]
        if not exact_candidates:
            st.warning(
                "No low-cardinality categorical covariates available for exact matching. "
                "Add categorical variables with few unique values to the covariate list."
            )
        exact_cols = st.multiselect(
            "Variables to match exactly",
            exact_candidates,
            default=exact_candidates[:min(2, len(exact_candidates))],
            key="ci_exact_cols",
        )
        if matching_method == "Exact + Propensity Score (Mixed)":
            with_replacement = st.checkbox(
                "Allow replacement within strata", value=False, key="ci_mixed_replacement",
            )

    # ── Caliper settings ───────────────────────────────────────
    st.markdown("### Caliper Settings")
    cal1, cal2 = st.columns(2)
    with cal1:
        caliper_mode = st.radio(
            "Caliper mode",
            ["Auto (0.2 x SD of logit(PS))", "Manual"],
            key="ci_caliper_mode",
        )
    with cal2:
        if caliper_mode == "Manual":
            caliper_manual = st.number_input(
                "Caliper (max PS distance)",
                min_value=0.001, max_value=0.5,
                value=0.2, step=0.01,
                key="ci_caliper_manual",
            )
        else:
            caliper_manual = None
            st.caption("Caliper will be computed automatically after propensity scores are estimated.")

    # ── Common support trimming ───────────────────────────────
    trim_support = st.checkbox(
        "Trim observations outside common support before matching",
        value=True, key="ci_trim_support",
        help="Recommended. Removes units with no realistic comparison in the other group.",
    )

    # ── IPW + extras ───────────────────────────────────────────
    st.markdown("### Additional Analyses")
    ax1, ax2, ax3 = st.columns(3)
    with ax1:
        run_ipw_also = st.checkbox("Run IPW (alongside matching)", value=True, key="ci_ipw")
    with ax2:
        run_sensitivity = st.checkbox("Run sensitivity analysis (Rosenbaum Bounds)", value=True, key="ci_sens")
    with ax3:
        run_ate = st.checkbox("Estimate ATE (in addition to ATT)", value=True, key="ci_ate")

    ax4, ax5 = st.columns(2)
    with ax4:
        run_subgroup = st.checkbox("Run subgroup analysis", value=False, key="ci_subgroup")
    with ax5:
        run_placebo = st.checkbox("Run placebo test", value=False, key="ci_placebo")

    # ── Guidance ──────────────────────────────────────────────
    with st.expander("How causal inference works here"):
        st.markdown("""
**Step 1 - Propensity Score:** Logistic regression estimates P(Treatment=1 | covariates).

**Step 2 - Matching:** Choose from 6 methods:
- Nearest Neighbor (1:1, 1:k, with replacement) - matches on propensity score
- Exact Matching - matches identically on selected categorical variables
- Exact + PSM (Mixed) - exact match on categories, then PSM within each group
- Mahalanobis Distance - matches on the full covariate vector, not just PS

**Step 3 - Outcome Analysis:** Regression on the matched sample gives the ATT.

**Additional analyses:**
- ATE: effect if treatment were applied to everyone (via IPW on full sample)
- Sensitivity (Rosenbaum Bounds): how robust the result is to hidden confounders
- Subgroup analysis: does the effect vary across population segments
- Placebo test: sanity check using an outcome that should show no effect
        """)

    # ── Run ───────────────────────────────────────────────────
    if st.button("Run Causal Analysis", use_container_width=True, key="ci_run"):
        if not covariate_cols:
            st.error("Select at least one covariate for the propensity score model.")
            return

        if matching_method in ("Exact Matching", "Exact + Propensity Score (Mixed)") and not exact_cols:
            st.error("Select at least one variable for exact matching.")
            return

        unique_t = mdf[treatment_col].dropna().unique()
        if len(unique_t) != 2:
            st.error(
                "Treatment variable must have exactly 2 unique values. "
                "Found: " + str(sorted(unique_t))
            )
            return

        if not set(unique_t).issubset({0, 1}):
            sorted_t = sorted(unique_t)
            mdf[treatment_col] = mdf[treatment_col].map(
                {sorted_t[0]: 0, sorted_t[1]: 1}
            )
            st.info("Treatment encoded: " + str(sorted_t[0]) + "->0, " + str(sorted_t[1]) + "->1")

        key_cols = list(set([treatment_col, outcome_col] + covariate_cols))
        data_clean = mdf[key_cols].dropna().copy().reset_index(drop=True)

        if len(data_clean) < 20:
            st.error("Too few complete observations (" + str(len(data_clean)) + "). Need at least 20.")
            return

        n_treated_orig = int((data_clean[treatment_col] == 1).sum())
        n_control_orig = int((data_clean[treatment_col] == 0).sum())
        st.markdown(
            "Complete observations: **" + str(len(data_clean)) + "** | "
            "Treated: **" + str(n_treated_orig) + "** | "
            "Control: **" + str(n_control_orig) + "**"
        )

        try:
            ci_progress = st.progress(0, text="Step 1/5: Estimating propensity scores...")

            # ── Step 1 ──────────────────────────────────────────
            st.markdown("---")
            ps, X_raw, ps_issues = step1_propensity_score(
                data_clean, treatment_col, covariate_cols, plot_template
            )

            auc_value = float(roc_auc_score(
                data_clean[treatment_col].astype(int), ps
            ))

            ci_progress.progress(20, text="Step 2/5: Checking common support...")

            # ── Common support trimming ─────────────────────────
            st.markdown("---")
            data_for_matching, ps_for_matching, support_summary = render_common_support_section(
                data_clean, ps, treatment_col, plot_template, trim_support,
            )

            if len(data_for_matching) < 20:
                ci_progress.empty()
                st.error("Too few observations remain after common support trimming.")
                return

            # ── Smart caliper ────────────────────────────────────
            auto_caliper, sd_logit = compute_smart_caliper(ps_for_matching)
            if caliper_mode == "Manual":
                caliper_value = float(caliper_manual)
            else:
                caliper_value = auto_caliper
                st.info(
                    "Auto caliper computed: " + str(auto_caliper) +
                    " (0.2 x SD of logit(PS) = " + str(sd_logit) + ")"
                )

            ci_progress.progress(40, text="Step 3/5: Matching treated and control units...")

            # ── Step 2: Matching ─────────────────────────────────
            st.markdown("---")
            # X_raw indices must align with data_for_matching after trimming
            if trim_support and support_summary["n_excluded"] > 0:
                T_full = data_clean[treatment_col].astype(int).values
                ps_treated_full = ps[T_full == 1]
                ps_control_full = ps[T_full == 0]
                overlap_min = max(ps_treated_full.min(), ps_control_full.min())
                overlap_max = min(ps_treated_full.max(), ps_control_full.max())
                in_support_mask = (ps >= overlap_min) & (ps <= overlap_max)
                X_raw_for_matching = X_raw.iloc[in_support_mask].reset_index(drop=True)
            else:
                X_raw_for_matching = X_raw

            matched_data, balance_df = step2_matching(
                data_for_matching, treatment_col, covariate_cols,
                ps_for_matching, X_raw_for_matching,
                matching_method, caliper_value, n_neighbors,
                exact_cols, with_replacement, plot_template,
            )

            if matched_data is None:
                ci_progress.empty()
                return

            n_matched = int((matched_data[treatment_col] == 1).sum())
            match_rate = round(n_matched / n_treated_orig * 100, 1) if n_treated_orig > 0 else 0

            ci_progress.progress(60, text="Step 4/5: Running outcome analysis...")

            # ── Step 3: Outcome ──────────────────────────────────
            st.markdown("---")
            outcome_results = step3_outcome(
                matched_data, treatment_col, outcome_col,
                covariate_cols, outcome_type, plot_template,
            )

            ate_result = None

            if outcome_results:
                # ── ATE comparison ─────────────────────────────
                if run_ate:
                    st.markdown("---")
                    render_att_vs_ate(
                        outcome_results["att_value"], outcome_results["att_pval"],
                        outcome_results["att_ci_lo"], outcome_results["att_ci_hi"],
                        data_clean, ps, treatment_col, outcome_col, outcome_type,
                    )
                    ate_result = estimate_ate(
                        data_clean, ps, treatment_col, outcome_col, outcome_type,
                    )

            ci_progress.progress(80, text="Step 5/5: Running additional analyses...")

            rosenbaum_df = None

            # ── Sensitivity analysis ────────────────────────────
            if run_sensitivity:
                st.markdown("---")
                bounds_df, _ = rosenbaum_bounds(
                    matched_data, treatment_col, outcome_col, outcome_type,
                )
                rosenbaum_df = bounds_df
                render_rosenbaum_section(
                    matched_data, treatment_col, outcome_col, outcome_type, plot_template,
                )

            # ── Subgroup analysis ────────────────────────────────
            if run_subgroup:
                st.markdown("---")
                render_subgroup_analysis(
                    matched_data, treatment_col, outcome_col,
                    outcome_type, all_cols, plot_template,
                )

            # ── Placebo test ─────────────────────────────────────
            if run_placebo:
                st.markdown("---")
                render_placebo_test(
                    matched_data, treatment_col, all_cols, outcome_col, plot_template,
                )

            # ── IPW ───────────────────────────────────────────────
            if run_ipw_also:
                st.markdown("---")
                run_ipw(
                    data_clean, treatment_col, outcome_col,
                    covariate_cols, ps, outcome_type, plot_template,
                )

            ci_progress.progress(100, text="Done!")
            ci_progress.empty()

            # ── Full report download ────────────────────────────
            if outcome_results:
                st.markdown("---")
                st.markdown("### Download Full Causal Report")
                html_report = generate_causal_html_report(
                    treatment_col, outcome_col, outcome_type, matching_method,
                    n_treated_orig, n_control_orig,
                    round(auc_value, 4), support_summary["pct_excluded"],
                    n_matched, match_rate, balance_df,
                    outcome_results["att_value"], outcome_results["att_pval"],
                    outcome_results["att_ci_lo"], outcome_results["att_ci_hi"],
                    ate_result,
                    rosenbaum_df,
                )
                st.download_button(
                    "📥 Download Causal Inference Report (HTML)",
                    data=html_report,
                    file_name="causal_inference_report.html",
                    mime="text/html",
                    use_container_width=True,
                )

        except Exception as e:
            st.error("Causal analysis error: " + str(e))
            with st.expander("Error details"):
                import traceback
                st.code(traceback.format_exc())
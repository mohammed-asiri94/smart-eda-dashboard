# ============================================================
# modules/models/causal.py
# Causal Inference: Propensity Score Matching + IPW
# Step 1: Propensity Score (Logistic)
# Step 2: Matching (Nearest Neighbor + Caliper)
# Step 3: Outcome Analysis (auto model selection)
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


# ============================================================
# Helpers
# ============================================================

def _show_issue(level, msg, fix=""):
    if level == "error":
        st.error("RED: " + msg + ("\n\nFix: " + fix if fix else ""))
    elif level == "warning":
        st.warning("YELLOW: " + msg + ("\n\nFix: " + fix if fix else ""))
    else:
        st.info("INFO: " + msg)


def _diagnostic_summary(issues):
    if not issues:
        st.success("All diagnostic checks passed.")
    else:
        for i in issues:
            _show_issue(i["level"], i["msg"], i.get("fix", ""))


def _quote(col):
    return 'Q("' + str(col).replace('"', '\\"') + '")'


def _smd(x1, x2):
    """Standardized Mean Difference between two groups."""
    mean1, mean2 = np.mean(x1), np.mean(x2)
    var1,  var2  = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_sd = np.sqrt((var1 + var2) / 2)
    if pooled_sd == 0:
        return 0.0
    return abs(mean1 - mean2) / pooled_sd


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
    st.markdown("### Step 1 Diagnostics")
    _diagnostic_summary(issues)

    return ps, X_raw, issues


# ============================================================
# Step 2: Matching
# ============================================================

def step2_matching(data, treatment_col, covariate_cols, ps,
                   X_raw, caliper, n_neighbors, plot_template):
    st.markdown("## Step 2: Propensity Score Matching")
    st.caption(
        "Each treated unit is matched to the most similar control unit "
        "based on propensity score distance."
    )

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

    # ── Nearest Neighbor Matching ─────────────────────────────
    ps_treated = ps[treated_idx].reshape(-1, 1)
    ps_control = ps[control_idx].reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(ps_control)
    distances, indices = nn.kneighbors(ps_treated)

    # Apply caliper
    matched_treated = []
    matched_control = []
    unmatched = 0

    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        valid = [(d, idx) for d, idx in zip(dists, idxs) if d <= caliper]
        if valid:
            best_idx = valid[0][1]
            matched_treated.append(treated_idx[i])
            matched_control.append(control_idx[best_idx])
        else:
            unmatched += 1

    if not matched_treated:
        st.error(
            "No matches found within caliper = " + str(caliper) + ". "
            "Try increasing the caliper."
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
    m4.metric("Caliper used",      caliper)

    if match_rate < 70:
        issues.append({
            "level": "warning",
            "msg": "Match rate is " + str(match_rate) + "% - many treated units unmatched.",
            "fix": "Increase caliper size or use IPW which uses all observations.",
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

        # Before matching
        before_t = data.loc[data[treatment_col] == 1, col].dropna()
        before_c = data.loc[data[treatment_col] == 0, col].dropna()
        smd_before = round(_smd(before_t.values, before_c.values), 4)

        # After matching
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

        # Love plot
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

        # Lines connecting before/after
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

        # Balance summary
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

    st.markdown("### Step 2 Diagnostics")
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
        "Download outcome model results (CSV)",
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

    st.markdown("### Step 3 Diagnostics")
    _diagnostic_summary(issues)

    with st.expander("Full Model Summary"):
        st.text(result.summary().as_text())


# ============================================================
# IPW — Inverse Probability Weighting
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

    st.markdown("### IPW Diagnostics")
    _diagnostic_summary(issues)


# ============================================================
# Main render function
# ============================================================

def render_causal_tab(df, df_cleaned, plot_template):
    st.markdown("# Causal Inference")
    st.info(
        "Estimates the **causal effect** of a treatment or intervention "
        "by balancing observed confounders between treated and control groups. "
        "Uses Propensity Score Matching (PSM) and/or Inverse Probability Weighting (IPW)."
    )

    # Dataset
    dataset_choice = st.radio(
        "Dataset to use",
        ["Original data", "Cleaned data (from Data Cleaning tab)"],
        horizontal=True,
        key="ci_dataset",
    )
    mdf = df_cleaned.copy() if dataset_choice.startswith("Cleaned") else df.copy()

    all_cols     = mdf.columns.tolist()
    numeric_cols = mdf.select_dtypes(include=np.number).columns.tolist()

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

    available_covs = [c for c in all_cols if c not in (treatment_col, outcome_col)]
    covariate_cols = st.multiselect(
        "Confounders / covariates for propensity score",
        available_covs,
        default=available_covs[:min(6, len(available_covs))],
        key="ci_covariates",
        help="Variables that affect both treatment assignment and outcome.",
    )

    # ── Matching settings ─────────────────────────────────────
    st.markdown("### Matching Settings")
    ms1, ms2, ms3 = st.columns(3)
    with ms1:
        caliper = st.number_input(
            "Caliper (max PS distance)",
            min_value=0.001, max_value=0.5,
            value=0.2, step=0.01,
            key="ci_caliper",
            help="Maximum allowed propensity score distance for a valid match. "
                 "Smaller = stricter matching. Typical: 0.1-0.25.",
        )
    with ms2:
        n_neighbors = st.number_input(
            "Neighbors to consider",
            min_value=1, max_value=10,
            value=1, step=1,
            key="ci_neighbors",
        )
    with ms3:
        run_ipw_also = st.checkbox(
            "Also run IPW",
            value=True,
            key="ci_ipw",
            help="Run IPW in addition to matching for comparison.",
        )

    # ── Guidance ──────────────────────────────────────────────
    with st.expander("How causal inference works here"):
        st.markdown("""
**Step 1 - Propensity Score:**
- Logistic regression: P(Treatment=1 | covariates)
- Check: AUC between 0.6-0.9, good overlap between groups

**Step 2 - Matching:**
- Each treated unit matched to nearest control by PS distance
- Caliper prevents bad matches
- Check: SMD < 0.1 for all covariates after matching (Love Plot)

**Step 3 - Outcome Analysis:**
- Regression on matched sample
- Treatment coefficient = ATT (Average Treatment Effect on the Treated)
- Model type chosen based on outcome type

**Key assumptions:**
- Ignorability: no unmeasured confounders
- Positivity: all units have PS between 0 and 1
- SUTVA: one unit's treatment does not affect another's outcome
        """)

    # ── Run ───────────────────────────────────────────────────
    if st.button("Run Causal Analysis", use_container_width=True, key="ci_run"):
        if not covariate_cols:
            st.error("Select at least one covariate for the propensity score model.")
            return

        # Validate treatment
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

        # Drop rows with missing in key columns
        key_cols = [treatment_col, outcome_col] + covariate_cols
        data_clean = mdf[key_cols].dropna().copy()

        if len(data_clean) < 20:
            st.error("Too few complete observations (" + str(len(data_clean)) + "). Need at least 20.")
            return

        n_treated = (data_clean[treatment_col] == 1).sum()
        n_control = (data_clean[treatment_col] == 0).sum()
        st.markdown(
            "Complete observations: **" + str(len(data_clean)) + "** | "
            "Treated: **" + str(n_treated) + "** | "
            "Control: **" + str(n_control) + "**"
        )

        try:
            # Step 1
            st.markdown("---")
            ps, X_raw, ps_issues = step1_propensity_score(
                data_clean, treatment_col, covariate_cols, plot_template
            )

            # Step 2
            st.markdown("---")
            matched_data, balance_df = step2_matching(
                data_clean, treatment_col, covariate_cols,
                ps, X_raw, float(caliper), int(n_neighbors), plot_template,
            )

            if matched_data is None:
                st.error("Matching failed. Try increasing the caliper.")
                return

            # Step 3
            st.markdown("---")
            step3_outcome(
                matched_data, treatment_col, outcome_col,
                covariate_cols, outcome_type, plot_template,
            )

            # IPW
            if run_ipw_also:
                st.markdown("---")
                run_ipw(
                    data_clean, treatment_col, outcome_col,
                    covariate_cols, ps, outcome_type, plot_template,
                )

        except Exception as e:
            st.error("Causal analysis error: " + str(e))
            with st.expander("Error details"):
                import traceback
                st.code(traceback.format_exc())
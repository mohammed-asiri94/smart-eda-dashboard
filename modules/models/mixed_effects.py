# ============================================================
# modules/models/mixed_effects.py
# Linear Mixed Effects (LME) + Generalized LME (GLME)
# Full diagnostics + fix suggestions on the same page
# All text ASCII only - no special characters
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
from statsmodels.stats.stattools import durbin_watson


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
        st.success("✅ All diagnostic checks passed - no major issues detected.")
    else:
        for i in issues:
            _show_issue(i["level"], i["msg"], i.get("fix", ""))


def _quote(col):
    """Quote column name for statsmodels formula."""
    return 'Q("' + str(col).replace('"', '\\"') + '")'


def _build_formula(outcome, fixed_effects, df):
    """Build statsmodels formula string."""
    cat_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    parts = []
    for col in fixed_effects:
        if col in cat_cols:
            parts.append("C(" + _quote(col) + ")")
        else:
            parts.append(_quote(col))
    return _quote(outcome) + " ~ " + " + ".join(parts) if parts else _quote(outcome) + " ~ 1"


# ============================================================
# ICC calculation
# ============================================================

def calculate_icc(model_result):
    """
    Calculate Intraclass Correlation Coefficient (ICC).
    ICC = variance_between / (variance_between + variance_within)
    """
    try:
        var_between = float(model_result.cov_re.iloc[0, 0])
        var_within  = float(model_result.scale)
        icc = var_between / (var_between + var_within)
        return round(icc, 4), var_between, var_within
    except Exception:
        return None, None, None


# ============================================================
# Linear Mixed Effects Model
# ============================================================

def run_lme(df, outcome, fixed_effects, group_col, random_slopes, plot_template):
    st.markdown("## Linear Mixed Effects Model (LME)")
    st.caption(
        "LME accounts for non-independence in clustered or repeated-measures data. "
        "Fixed effects = population-level estimates. "
        "Random effects = group-level variation."
    )

    issues = []

    # ── Prepare data ──────────────────────────────────────────
    cols = [outcome, group_col] + fixed_effects
    cols = list(dict.fromkeys(cols))  # remove duplicates
    data = df[cols].dropna().copy()

    if data.empty:
        st.error("No data remaining after dropping missing values.")
        return

    n_total  = len(data)
    n_groups = data[group_col].nunique()
    n_per_group = data.groupby(group_col).size()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations",    f"{n_total:,}")
    c2.metric("Groups",          f"{n_groups:,}")
    c3.metric("Min obs/group",   int(n_per_group.min()))
    c4.metric("Max obs/group",   int(n_per_group.max()))

    # ── Pre-fit checks ────────────────────────────────────────
    if n_groups < 5:
        issues.append({
            "level": "error",
            "msg": f"Only {n_groups} groups detected. Mixed effects models need at least 5-10 groups.",
            "fix": "With very few groups, consider fixed effects with dummy variables instead.",
        })

    if n_per_group.min() < 2:
        issues.append({
            "level": "warning",
            "msg": "Some groups have only 1 observation - random effects cannot be estimated for them.",
            "fix": "Remove groups with fewer than 2 observations or merge small groups.",
        })

    # ── Build formula ─────────────────────────────────────────
    formula = _build_formula(outcome, fixed_effects, data)

    # Random effects formula
    if random_slopes:
        re_formula = "~" + " + ".join([_quote(c) for c in random_slopes])
    else:
        re_formula = None  # random intercept only

    # ── Fit model ─────────────────────────────────────────────
    try:
        with st.spinner("Fitting Linear Mixed Effects model..."):
            model = smf.mixedlm(
                formula=formula,
                data=data,
                groups=data[group_col],
                re_formula=re_formula,
            )
            result = model.fit(reml=True, method="lbfgs")

    except Exception as e:
        st.error("Model failed to converge: " + str(e))
        issues.append({
            "level": "error",
            "msg": "Model convergence failed.",
            "fix": (
                "Try: (1) Remove random slopes and use random intercept only. "
                "(2) Standardize numeric predictors. "
                "(3) Reduce number of fixed effects. "
                "(4) Use REML=False for complex models."
            ),
        })
        _diagnostic_summary(issues)
        return

    st.success("Model fitted successfully.")

    # ── ICC ───────────────────────────────────────────────────
    icc, var_between, var_within = calculate_icc(result)

    st.markdown("### Intraclass Correlation (ICC)")
    if icc is not None:
        ic1, ic2, ic3 = st.columns(3)
        ic1.metric("ICC", icc,
                   help="Proportion of total variance explained by group membership.")
        ic2.metric("Between-group variance", round(var_between, 4))
        ic3.metric("Within-group variance",  round(var_within, 4))

        # ICC interpretation
        if icc < 0.05:
            st.info(
                "ICC = " + str(icc) + " - Very low group-level variance. "
                "A simple OLS regression may be sufficient."
            )
            issues.append({
                "level": "warning",
                "msg": "ICC is very low (" + str(icc) + ") - mixed effects may not be necessary.",
                "fix": "Consider running a standard Linear Regression instead. "
                       "Run LR Test below to confirm.",
            })
        elif icc < 0.10:
            st.warning("ICC = " + str(icc) + " - Low but non-trivial group-level variance.")
        elif icc < 0.30:
            st.success("ICC = " + str(icc) + " - Moderate group-level variance. Mixed effects appropriate.")
        else:
            st.success("ICC = " + str(icc) + " - High group-level variance. Mixed effects strongly justified.")

    # ── Fixed effects table ───────────────────────────────────
    st.markdown("### Fixed Effects Coefficients")
    st.caption("Population-level estimates - apply to all groups on average.")

    params = list(result.params)
    pvals  = list(result.pvalues)
    bse    = list(result.bse)
    conf   = result.conf_int()
    ci_lo  = list(conf[0])
    ci_hi  = list(conf[1])
    names  = list(result.params.index)

    coef_tbl = pd.DataFrame({
        "Term":        names,
        "Coefficient": [round(float(x), 4) for x in params],
        "Std Error":   [round(float(x), 4) for x in bse],
        "p-value":     [round(float(x), 5) for x in pvals],
        "CI Lower":    [round(float(x), 4) for x in ci_lo],
        "CI Upper":    [round(float(x), 4) for x in ci_hi],
        "Significant": ["YES" if float(p) < 0.05 else "NO" for p in pvals],
    })
    st.dataframe(coef_tbl, use_container_width=True)

    st.download_button(
        "📥 Download fixed effects (CSV)",
        data=coef_tbl.to_csv(index=False).encode(),
        file_name="lme_fixed_effects.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Forest plot ───────────────────────────────────────────
    st.markdown("### Forest Plot - Fixed Effects")
    fe_plot = coef_tbl[coef_tbl["Term"] != "Intercept"].copy()
    if not fe_plot.empty:
        fig_forest = go.Figure()
        fig_forest.add_trace(go.Scatter(
            x=fe_plot["Coefficient"],
            y=fe_plot["Term"],
            mode="markers",
            marker=dict(size=10, color="#2563EB"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=(fe_plot["CI Upper"] - fe_plot["Coefficient"]).tolist(),
                arrayminus=(fe_plot["Coefficient"] - fe_plot["CI Lower"]).tolist(),
            ),
            name="Coefficient (95% CI)",
        ))
        fig_forest.add_vline(x=0, line_dash="dash", line_color="red",
                             annotation_text="No effect")
        fig_forest.update_layout(
            title="Fixed Effects Forest Plot",
            xaxis_title="Coefficient",
            template=plot_template,
            height=max(300, len(fe_plot) * 45 + 100),
        )
        st.plotly_chart(fig_forest, use_container_width=True)

    # ── Random effects ────────────────────────────────────────
    st.markdown("### Random Effects by Group")
    st.caption("Group-specific deviations from the population-level intercept.")

    try:
        re = result.random_effects
        re_df = pd.DataFrame(
            {group_col: list(re.keys()),
             "Random Intercept": [float(v.iloc[0]) for v in re.values()]}
        ).sort_values("Random Intercept", ascending=True)

        fig_re = go.Figure()
        colors = ["#DC2626" if x < 0 else "#16A34A" for x in re_df["Random Intercept"]]
        fig_re.add_trace(go.Bar(
            x=re_df["Random Intercept"],
            y=re_df[group_col].astype(str),
            orientation="h",
            marker_color=colors,
            name="Random Intercept",
        ))
        fig_re.add_vline(x=0, line_dash="dash", line_color="grey")
        fig_re.update_layout(
            title="Random Effects by Group",
            xaxis_title="Random Intercept Deviation",
            yaxis_title=group_col,
            template=plot_template,
            height=max(300, len(re_df) * 25 + 100),
        )
        st.plotly_chart(fig_re, use_container_width=True)

        with st.expander("Random Effects Table"):
            st.dataframe(re_df, use_container_width=True)

    except Exception as e:
        st.info("Random effects plot could not be generated: " + str(e))

    # ── Model fit metrics ─────────────────────────────────────
    st.markdown("### Model Fit")
    m1, m2, m3 = st.columns(3)
    m1.metric("Log-Likelihood", round(result.llf, 3))
    m2.metric("AIC", round(result.aic, 3))
    m3.metric("BIC", round(result.bic, 3))

    # ── LR Test vs OLS ────────────────────────────────────────
    st.markdown("### Likelihood Ratio Test - Mixed vs Fixed Effects")
    st.caption(
        "Tests whether the random effects significantly improve model fit. "
        "p < 0.05 means mixed effects are justified."
    )
    try:
        ols_model  = smf.ols(formula=formula, data=data).fit()
        lr_stat    = 2 * (result.llf - ols_model.llf)
        lr_pval    = 1 - stats.chi2.cdf(lr_stat, df=1)

        lr1, lr2, lr3 = st.columns(3)
        lr1.metric("LR Statistic", round(lr_stat, 4))
        lr2.metric("p-value",      round(lr_pval, 5))
        lr3.metric("Result",
                   "Mixed effects justified (p<0.05)"
                   if lr_pval < 0.05
                   else "OLS may be sufficient (p>=0.05)")

        if lr_pval >= 0.05:
            issues.append({
                "level": "warning",
                "msg": "LR Test: random effects do not significantly improve fit (p=" + str(round(lr_pval,4)) + ").",
                "fix": "Consider using standard OLS regression instead of mixed effects.",
            })
        else:
            st.success("Mixed effects model significantly better than OLS (p=" + str(round(lr_pval,4)) + ").")

    except Exception as e:
        st.info("LR Test could not be performed: " + str(e))

    # ── Residual diagnostics ──────────────────────────────────
    st.markdown("### 🔬 Residual Diagnostics")
    try:
        residuals = result.resid
        fitted    = result.fittedvalues

        col_a, col_b = st.columns(2)

        with col_a:
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=fitted, y=residuals,
                mode="markers",
                marker=dict(size=4, color="#2563EB", opacity=0.5),
                name="Residuals",
            ))
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            fig_res.update_layout(
                title="Residuals vs Fitted",
                xaxis_title="Fitted Values",
                yaxis_title="Residuals",
                template=plot_template,
            )
            st.plotly_chart(fig_res, use_container_width=True)

        with col_b:
            std_resid = (residuals - residuals.mean()) / residuals.std()
            osm, osr  = stats.probplot(std_resid)[0]
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(
                x=list(osm), y=list(osr),
                mode="markers",
                marker=dict(size=4, color="#2563EB"),
                name="Residuals",
            ))
            fig_qq.add_trace(go.Scatter(
                x=[min(osm), max(osm)],
                y=[min(osm), max(osm)],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Normal line",
            ))
            fig_qq.update_layout(
                title="Q-Q Plot of Residuals",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                template=plot_template,
            )
            st.plotly_chart(fig_qq, use_container_width=True)

        # Normality test
        if len(residuals) <= 5000:
            _, sw_p = stats.shapiro(residuals)
            st.metric("Shapiro-Wilk p-value", round(sw_p, 4),
                      help="p > 0.05 means residuals are approximately normal.")
            if sw_p < 0.05:
                issues.append({
                    "level": "warning",
                    "msg": "Residuals may not be normally distributed (Shapiro-Wilk p=" + str(round(sw_p,4)) + ").",
                    "fix": "With large samples this is less critical. "
                           "Consider log-transforming the outcome variable.",
                })

        # Residuals by group
        st.markdown("#### Residuals by Group")
        resid_df = pd.DataFrame({
            "Group":    data[group_col].astype(str).values,
            "Residual": residuals.values,
        })
        fig_rg = px.box(
            resid_df, x="Group", y="Residual",
            title="Residuals by Group",
            template=plot_template,
        )
        fig_rg.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_rg, use_container_width=True)
        st.caption(
            "Residuals should be roughly centered at 0 for each group. "
            "Systematic patterns suggest model misspecification."
        )

    except Exception as e:
        st.info("Residual diagnostics could not be fully generated: " + str(e))

    # ── Diagnostic summary ────────────────────────────────────
    st.markdown("### 🩺 Diagnostic Summary")
    _diagnostic_summary(issues)

    with st.expander("Full Model Summary"):
        st.text(result.summary().as_text())


# ============================================================
# Generalized Linear Mixed Effects (GLME)
# ============================================================

def run_glme(df, outcome, fixed_effects, group_col, family_name, plot_template):
    st.markdown("## Generalized Linear Mixed Effects (GLME)")
    st.caption(
        "GLME extends LME to non-normal outcomes: "
        "binary (Logistic), counts (Poisson), etc."
    )

    issues = []

    cols = [outcome, group_col] + fixed_effects
    cols = list(dict.fromkeys(cols))
    data = df[cols].dropna().copy()

    if data.empty:
        st.error("No data remaining after dropping missing values.")
        return

    n_total  = len(data)
    n_groups = data[group_col].nunique()

    c1, c2 = st.columns(2)
    c1.metric("Observations", f"{n_total:,}")
    c2.metric("Groups",       f"{n_groups:,}")

    if n_groups < 5:
        issues.append({
            "level": "error",
            "msg": "Very few groups (" + str(n_groups) + "). GLME requires at least 5-10 groups.",
            "fix": "Use standard GLM instead or collect data from more groups.",
        })

    # Family mapping
    family_map = {
        "Logistic (binary outcome)":   sm.families.Binomial(),
        "Poisson (count outcome)":     sm.families.Poisson(),
        "Negative Binomial (overdispersed count)": sm.families.NegativeBinomial(),
    }
    family = family_map.get(family_name, sm.families.Gaussian())

    # Validate outcome
    if "Logistic" in family_name:
        unique_vals = data[outcome].dropna().unique()
        if not set(unique_vals).issubset({0, 1, True, False}):
            sorted_cls = sorted(unique_vals)
            data[outcome] = data[outcome].map({sorted_cls[0]: 0, sorted_cls[1]: 1})
            st.info("Outcome encoded: " + str(sorted_cls[0]) + " -> 0, " + str(sorted_cls[1]) + " -> 1")

    formula = _build_formula(outcome, fixed_effects, data)

    try:
        with st.spinner("Fitting GLME model..."):
            model = smf.glm(
                formula=formula,
                data=data,
                groups=data[group_col],
                family=family,
            )
            result = model.fit()

    except Exception as e:
        # fallback: try without groups (standard GLM)
        st.warning(
            "GLME with random effects failed. Fitting standard GLM instead. "
            "Error: " + str(e)
        )
        try:
            result = smf.glm(
                formula=formula,
                data=data,
                family=family,
            ).fit()
            st.info("Showing standard GLM results (no random effects).")
        except Exception as e2:
            st.error("Model failed: " + str(e2))
            return

    st.success("Model fitted successfully.")

    c1, c2, c3 = st.columns(3)
    c1.metric("N",         int(result.nobs))
    c2.metric("AIC",       round(result.aic, 2))
    c3.metric("Deviance",  round(result.deviance, 2))

    # Coefficients
    st.markdown("### Coefficients")
    params = list(result.params)
    pvals  = list(result.pvalues)
    bse    = list(result.bse)
    conf   = result.conf_int()
    ci_lo  = list(conf[0])
    ci_hi  = list(conf[1])
    names  = list(result.params.index)

    coef_tbl = pd.DataFrame({
        "Term":        names,
        "Coefficient": [round(float(x), 4) for x in params],
        "Std Error":   [round(float(x), 4) for x in bse],
        "p-value":     [round(float(x), 5) for x in pvals],
        "CI Lower":    [round(float(x), 4) for x in ci_lo],
        "CI Upper":    [round(float(x), 4) for x in ci_hi],
    })

    if "Logistic" in family_name:
        coef_tbl["Odds Ratio"] = [round(float(np.exp(x)), 4) for x in params]

    if "Poisson" in family_name or "Binomial" in family_name:
        coef_tbl["IRR"] = [round(float(np.exp(x)), 4) for x in params]

    coef_tbl["Significant"] = ["YES" if float(p) < 0.05 else "NO" for p in pvals]
    st.dataframe(coef_tbl, use_container_width=True)

    st.download_button(
        "📥 Download GLME coefficients (CSV)",
        data=coef_tbl.to_csv(index=False).encode(),
        file_name="glme_coefficients.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Diagnostics
    st.markdown("### 🔬 Diagnostics")
    disp = result.pearson_chi2 / result.df_resid
    st.metric("Pearson chi2 / df", round(disp, 4),
              help="Value near 1.0 = good fit. >> 1 = overdispersion.")
    if disp > 1.5:
        issues.append({
            "level": "error",
            "msg": "Overdispersion detected (Pearson chi2/df = " + str(round(disp,3)) + ").",
            "fix": "Switch to Negative Binomial family for count outcomes.",
        })

    st.markdown("### 🩺 Diagnostic Summary")
    _diagnostic_summary(issues)

    with st.expander("Full Model Summary"):
        st.text(result.summary().as_text())


# ============================================================
# Main render function
# ============================================================

def render_mixed_effects_tab(df, df_cleaned, plot_template):
    st.markdown("# Mixed Effects Models")
    st.info(
        "Mixed effects models handle **clustered, hierarchical, or repeated-measures data** "
        "where observations within groups are not independent. "
        "Examples: patients within hospitals, students within schools, "
        "repeated measurements on the same subject."
    )

    # Dataset choice
    dataset_choice = st.radio(
        "Dataset to use",
        ["Original data", "Cleaned data (from Data Cleaning tab)"],
        horizontal=True,
        key="me_dataset",
    )
    mdf = df_cleaned.copy() if dataset_choice.startswith("Cleaned") else df.copy()

    all_cols     = mdf.columns.tolist()
    numeric_cols = mdf.select_dtypes(include=np.number).columns.tolist()

    if len(all_cols) < 3:
        st.warning("Need at least 3 columns: outcome + grouping + at least one predictor.")
        return

    # ── Column setup ──────────────────────────────────────────
    st.markdown("### Column Setup")
    me1, me2, me3 = st.columns(3)

    with me1:
        outcome = st.selectbox(
            "Outcome variable (Y)",
            all_cols,
            key="me_outcome",
            help="The variable you want to predict.",
        )
    with me2:
        group_options = [c for c in all_cols if c != outcome]
        group_col = st.selectbox(
            "Grouping variable (cluster)",
            group_options,
            key="me_group",
            help="The variable that defines groups (e.g., hospital, school, subject ID).",
        )
    with me3:
        available_fe = [c for c in all_cols if c not in (outcome, group_col)]
        fixed_effects = st.multiselect(
            "Fixed effects (predictors)",
            available_fe,
            default=available_fe[:min(4, len(available_fe))],
            key="me_fixed",
        )

    # ── Model type ────────────────────────────────────────────
    st.markdown("### Model Type")
    model_type = st.radio(
        "Choose model",
        [
            "Linear Mixed Effects (LME) - continuous outcome",
            "Generalized LME (GLME) - binary or count outcome",
        ],
        horizontal=True,
        key="me_model_type",
    )

    # ── Additional options ────────────────────────────────────
    random_slopes = []
    family_name   = "Logistic (binary outcome)"

    if "LME" in model_type and not "Generalized" in model_type:
        st.markdown("### Random Effects Structure")
        num_fixed = [c for c in fixed_effects
                     if pd.api.types.is_numeric_dtype(mdf[c])]
        if num_fixed:
            random_slopes = st.multiselect(
                "Add random slopes for (optional):",
                num_fixed,
                key="me_random_slopes",
                help="Leave empty for random intercept only (recommended to start).",
            )
            if random_slopes:
                st.caption(
                    "Random slopes allow the effect of a predictor to vary across groups. "
                    "Start with random intercept only if model fails to converge."
                )
    else:
        family_name = st.selectbox(
            "Family / link function",
            [
                "Logistic (binary outcome)",
                "Poisson (count outcome)",
                "Negative Binomial (overdispersed count)",
            ],
            key="me_family",
        )

    # ── Guidance ──────────────────────────────────────────────
    with st.expander("When to use Mixed Effects Models"):
        st.markdown("""
**Use LME when:**
- Outcome is continuous (e.g. length of stay, blood pressure, test score)
- Data is clustered (patients in hospitals, employees in companies)
- You have repeated measurements on the same subject

**Use GLME when:**
- Outcome is binary (Yes/No, 0/1) - use Logistic family
- Outcome is a count (number of events) - use Poisson or NB family
- Data is still clustered or repeated

**Key concepts:**
- **ICC (Intraclass Correlation):** How much of the total variance is due to group membership
  - ICC > 0.05 generally justifies mixed effects
- **Fixed effects:** Average effect across all groups
- **Random effects:** How much each group deviates from the average
- **LR Test:** Confirms whether mixed effects significantly improve fit over OLS
        """)

    # ── Run button ────────────────────────────────────────────
    if st.button("Run Mixed Effects Model", use_container_width=True, key="me_run"):
        if not fixed_effects:
            st.error("Select at least one fixed effect predictor.")
            return
        try:
            if "Generalized" in model_type:
                run_glme(
                    mdf, outcome, fixed_effects,
                    group_col, family_name, plot_template,
                )
            else:
                run_lme(
                    mdf, outcome, fixed_effects,
                    group_col, random_slopes, plot_template,
                )
        except Exception as e:
            st.error("Mixed effects model error: " + str(e))
            with st.expander("Error details"):
                import traceback
                st.code(traceback.format_exc())
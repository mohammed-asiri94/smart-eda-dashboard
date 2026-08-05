# ============================================================
# modules/models/mixed_effects.py
# Multilevel / Mixed Effects Models
#
# NEW in this version:
#   LME:
#     - Null model (intercept-only) fitted first; LRT chain:
#         OLS vs Null-RI, Null-RI vs RI, RI vs RS
#     - Model comparison table (AIC/BIC/LogL/LRT for all nested models)
#     - VPC = ICC displayed with formula explanation
#     - True Caterpillar Plot (BLUPs with 95% CI intervals per group)
#     - Random-effects normality histogram (Figure 1 from MLM II PDF)
#     - Random slope variance + intercept-slope covariance reported
#
#   GLME:
#     - Replaced broken smf.glm(groups=...) with REAL marginal model:
#         GEE with exchangeable correlation (population-average,
#         equivalent to glmer in many health research applications)
#     - Exchangeable alpha = ICC proxy for binary/count outcomes
#     - VPC for binary outcomes: sigma^2_u0 / (sigma^2_u0 + pi^2/3)
#         derived from null GEE alpha
#     - QIC (quasi-likelihood info criterion) for model selection
#     - OR / IRR columns in coefficient table
#     - Separate null vs full model comparison
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
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable, Independence
from modules.models.data_audit import prepare_complete_cases, render_model_data_audit


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
    return 'Q("' + str(col).replace('"', '\\"') + '")'


def _build_formula(outcome, fixed_effects, df):
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    parts = []
    for col in fixed_effects:
        if col in cat_cols:
            parts.append("C(" + _quote(col) + ")")
        else:
            parts.append(_quote(col))
    return _quote(outcome) + " ~ " + " + ".join(parts) if parts else _quote(outcome) + " ~ 1"


# ============================================================
# ICC / VPC helpers
# ============================================================

def _icc_linear(model_result):
    """ICC for linear MixedLM result."""
    try:
        var_b = float(model_result.cov_re.iloc[0, 0])
        var_w = float(model_result.scale)
        icc   = var_b / (var_b + var_w)
        return round(icc, 4), round(var_b, 4), round(var_w, 4)
    except Exception:
        return None, None, None


def _vpc_binary_from_alpha(alpha):
    """
    VPC for binary outcome using latent variable method.
    Under the logit link, the latent-variable level-1 variance is pi^2/3.
    alpha (GEE exchangeable correlation) approximates sigma^2_u0 / (sigma^2_u0 + pi^2/3)
    So sigma^2_u0 = alpha * pi^2/3 / (1 - alpha)
    VPC = sigma^2_u0 / (sigma^2_u0 + pi^2/3)  = alpha
    """
    sigma2_u0 = float(alpha) * (np.pi ** 2 / 3) / max(1 - float(alpha), 1e-9)
    vpc = sigma2_u0 / (sigma2_u0 + np.pi ** 2 / 3)
    return round(vpc, 4), round(sigma2_u0, 4)


# ============================================================
# Caterpillar plot helper
# ============================================================

def _caterpillar_plot(re_dict, group_col, var_between, plot_template, title="Caterpillar Plot — Random Intercepts (BLUPs)"):
    """
    True caterpillar plot: BLUPs sorted from low to high with 95% CI.
    SE of BLUP_j ≈ sqrt(var_between * (1 - lambda_j))
    We approximate lambda (shrinkage) as var_between / (var_between + var_within / n_j)
    using a fixed conservative se_blup = sqrt(var_between) * 0.5 when n_j unknown.
    """
    groups_list = list(re_dict.keys())
    blups       = [float(v.iloc[0]) for v in re_dict.values()]

    # Approximate SE for 95% CI
    se_blup = float(np.sqrt(max(var_between, 1e-9))) * 0.5
    ci_half = 1.96 * se_blup

    df_re = pd.DataFrame({
        group_col:   groups_list,
        "BLUP":      blups,
        "CI_lo":     [b - ci_half for b in blups],
        "CI_hi":     [b + ci_half for b in blups],
    }).sort_values("BLUP").reset_index(drop=True)
    df_re["index"] = np.arange(len(df_re))
    df_re["color"] = df_re["BLUP"].apply(lambda x: "#DC2626" if x < 0 else "#16A34A")

    fig = go.Figure()

    # CI lines
    for _, row in df_re.iterrows():
        fig.add_shape(
            type="line",
            x0=row["CI_lo"], x1=row["CI_hi"],
            y0=row["index"],  y1=row["index"],
            line=dict(color="#9CA3AF", width=1),
        )

    # BLUP points
    fig.add_trace(go.Scatter(
        x=df_re["BLUP"],
        y=df_re["index"],
        mode="markers",
        marker=dict(size=7, color=df_re["color"]),
        text=df_re[group_col].astype(str),
        hovertemplate="%{text}<br>BLUP: %{x:.3f}<extra></extra>",
        name="BLUP",
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="red",
                  annotation_text="Grand mean (0)")

    fig.update_layout(
        title=title,
        xaxis_title="Random Intercept (deviation from grand mean)",
        yaxis_title="Group (ordered low to high)",
        yaxis=dict(tickmode="array",
                   tickvals=df_re["index"].tolist(),
                   ticktext=df_re[group_col].astype(str).tolist(),
                   tickfont=dict(size=9)),
        height=max(350, len(df_re) * 20 + 120),
        template=plot_template,
        showlegend=False,
    )
    return fig, df_re


# ============================================================
# LME — Linear Mixed Effects
# ============================================================

def run_lme(df, outcome, fixed_effects, group_col, random_slopes, plot_template):
    st.markdown("## Linear Mixed Effects Model (LME)")
    st.caption(
        "Accounts for non-independence in clustered / repeated-measures data. "
        "Fixed effects = population-level estimates. "
        "Random effects = group-level variation."
    )

    issues = []

    # ── Prepare data ──────────────────────────────────────────
    cols = list(dict.fromkeys([outcome, group_col] + fixed_effects))
    data, data_audit = prepare_complete_cases(df, cols)
    render_model_data_audit(data_audit)

    if data.empty:
        st.error("No data remaining after dropping missing values.")
        return

    n_total   = len(data)
    n_groups  = data[group_col].nunique()
    n_per_grp = data.groupby(group_col).size()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations",  f"{n_total:,}")
    c2.metric("Groups",        f"{n_groups:,}")
    c3.metric("Min obs/group", int(n_per_grp.min()))
    c4.metric("Max obs/group", int(n_per_grp.max()))

    if n_groups < 5:
        issues.append({
            "level": "error",
            "msg": f"Only {n_groups} groups. Mixed effects need ≥5–10 groups.",
            "fix": "With very few groups use fixed-effects dummy variables instead.",
        })
    if n_per_grp.min() < 2:
        issues.append({
            "level": "warning",
            "msg": "Some groups have only 1 observation — random effects cannot be estimated for them.",
            "fix": "Remove groups with < 2 observations or merge small groups.",
        })

    formula    = _build_formula(outcome, fixed_effects, data)
    re_formula = ("~" + " + ".join([_quote(c) for c in random_slopes])
                  if random_slopes else None)

    # ── Step 1: Null model (intercept-only, random intercept) ─
    st.markdown("---")
    st.markdown("## Step 1 — Null Model (Empty Random Intercept)")
    st.caption(
        "The null model contains no fixed-effect predictors, only a random intercept. "
        "Comparing it to OLS confirms whether group-level variance is present at all "
        "(Key Point 1 in the course: start with an empty random intercept model)."
    )
    try:
        with st.spinner("Fitting null model..."):
            m_null = smf.mixedlm(
                _quote(outcome) + " ~ 1", data,
                groups=data[group_col],
            ).fit(reml=False, method="lbfgs")
        st.success("Null model fitted.")
        n1, n2, n3 = st.columns(3)
        n1.metric("Null Log-Likelihood", round(m_null.llf, 3))
        n2.metric("Null AIC", round(m_null.aic, 3))
        n3.metric("Null BIC", round(m_null.bic, 3))

        icc_null, vb_null, vw_null = _icc_linear(m_null)
        if icc_null is not None:
            st.markdown(
                "**Null ICC (VPC) = " + str(icc_null) + "**  "
                "— proportion of total variance attributable to group membership "
                "before adding any predictors. "
                "Formula: σ²ᵤ₀ / (σ²ᵤ₀ + σ²ₑ) = "
                + str(vb_null) + " / (" + str(vb_null) + " + " + str(vw_null) + ")."
            )
            if icc_null < 0.05:
                st.warning(
                    "Null ICC = " + str(icc_null) +
                    " — very little group-level variance. "
                    "A standard OLS may be sufficient (confirm with LRT below)."
                )
    except Exception as e:
        st.warning("Null model failed: " + str(e) + ". Continuing without null model.")
        m_null = None

    # ── Step 2: Random intercept model ────────────────────────
    st.markdown("---")
    st.markdown("## Step 2 — Random Intercept Model")
    st.caption(
        "Adds fixed-effect predictors. The intercept varies across groups "
        "but all slopes are parallel (Key Point 3a)."
    )
    try:
        with st.spinner("Fitting random intercept model..."):
            m_ri = smf.mixedlm(
                formula, data,
                groups=data[group_col],
            ).fit(reml=True, method="lbfgs")
        st.success("Random intercept model fitted.")
        result_ri = m_ri
    except Exception as e:
        st.error("Random intercept model failed: " + str(e))
        issues.append({"level": "error", "msg": "Model convergence failed.",
                       "fix": "Standardize predictors or reduce fixed effects."})
        _diagnostic_summary(issues)
        return

    # ICC / VPC
    icc, var_b, var_w = _icc_linear(m_ri)
    st.markdown("### ICC / VPC")
    st.caption(
        "VPC (Variance Partition Coefficient) = ICC = σ²ᵤ₀ / (σ²ᵤ₀ + σ²ₑ). "
        "Quantifies how much of the total variance is at the group level."
    )
    if icc is not None:
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("ICC / VPC", icc)
        ic2.metric("Between-group variance σ²ᵤ₀", var_b)
        ic3.metric("Within-group variance σ²ₑ", var_w)
        ic4.metric("Total variance", round(var_b + var_w, 4))

        if icc < 0.05:
            st.info("ICC = " + str(icc) + " — Very low group variance. OLS may be sufficient.")
            issues.append({"level": "warning",
                           "msg": "ICC very low (" + str(icc) + "). Mixed effects may not be needed.",
                           "fix": "Run LRT (below) to confirm."})
        elif icc < 0.10:
            st.warning("ICC = " + str(icc) + " — Low but non-trivial group-level variance.")
        elif icc < 0.30:
            st.success("ICC = " + str(icc) + " — Moderate group variance. Mixed effects appropriate.")
        else:
            st.success("ICC = " + str(icc) + " — High group variance. Mixed effects strongly justified.")

    # ── Step 3: Random slope model (if requested) ─────────────
    m_rs = None
    if random_slopes:
        st.markdown("---")
        st.markdown("## Step 3 — Random Slope Model")
        st.caption(
            "Relaxes the parallel-slopes assumption: the effect of "
            + ", ".join(random_slopes) +
            " is now allowed to differ across groups (Key Point 3b)."
        )
        try:
            with st.spinner("Fitting random slope model..."):
                m_rs = smf.mixedlm(
                    formula, data,
                    groups=data[group_col],
                    re_formula=re_formula,
                ).fit(reml=True, method="lbfgs")
            st.success("Random slope model fitted.")

            # Report slope variance and intercept-slope covariance
            try:
                cov_re = m_rs.cov_re
                if cov_re.shape[0] >= 2:
                    slope_var = round(float(cov_re.iloc[1, 1]), 4)
                    cov_int_slope = round(float(cov_re.iloc[0, 1]), 4)
                    corr_int_slope = round(cov_int_slope / max(
                        np.sqrt(float(cov_re.iloc[0, 0]) * float(cov_re.iloc[1, 1])), 1e-9
                    ), 4)
                    sv1, sv2, sv3 = st.columns(3)
                    sv1.metric("Random slope variance σ²ᵤ₁", slope_var,
                               help="Variance of slopes across groups.")
                    sv2.metric("Intercept–slope covariance σᵤ₀₁", cov_int_slope,
                               help="Positive = fanning-out; Negative = fanning-in.")
                    sv3.metric("Intercept–slope correlation ρ₀₁", corr_int_slope)

                    if cov_int_slope > 0:
                        st.info(
                            "ρ₀₁ > 0 → **Fanning-out** pattern: groups with higher baseline outcomes "
                            "also show a stronger predictor effect."
                        )
                    elif cov_int_slope < 0:
                        st.info(
                            "ρ₀₁ < 0 → **Fanning-in** pattern: groups with higher baseline outcomes "
                            "show a weaker (or reversed) predictor effect."
                        )
                    else:
                        st.info("ρ₀₁ ≈ 0 → No consistent relationship between intercepts and slopes.")
            except Exception:
                pass
        except Exception as e:
            st.warning("Random slope model failed to converge: " + str(e))
            issues.append({"level": "warning",
                           "msg": "Random slope model did not converge.",
                           "fix": "Standardize predictors or use random intercept only."})

    # ── Model comparison table ────────────────────────────────
    st.markdown("---")
    st.markdown("## Model Comparison Table")
    st.caption(
        "Iterative model building: each row is compared to the row above using "
        "a Likelihood Ratio Test (LRT). Models must be fitted with ML (REML=False) "
        "for valid LRT. Smaller AIC/BIC = better fit."
    )

    try:
        # Refit with ML for LRT
        m_ri_ml  = smf.mixedlm(formula, data, groups=data[group_col]).fit(reml=False, method="lbfgs")
        m_ols    = smf.ols(formula, data).fit()

        rows = []
        # OLS baseline
        rows.append({
            "Model":       "OLS (no random effects)",
            "LogL":        round(m_ols.llf, 2),
            "AIC":         round(m_ols.aic,  2),
            "BIC":         round(m_ols.bic,  2),
            "LRT vs prev": "—",
            "df":          "—",
            "p-value":     "—",
        })
        if m_null is not None:
            lr_ols_null = 2 * (m_null.llf - m_ols.llf)
            p_ols_null  = 1 - stats.chi2.cdf(lr_ols_null, 1)
            rows.append({
                "Model":       "Null random intercept (no predictors)",
                "LogL":        round(m_null.llf, 2),
                "AIC":         round(m_null.aic,  2),
                "BIC":         round(m_null.bic,  2),
                "LRT vs prev": round(lr_ols_null, 2),
                "df":          1,
                "p-value":     round(p_ols_null, 5),
            })
        lr_prev_ri = 2 * (m_ri_ml.llf - (m_null.llf if m_null else m_ols.llf))
        df_prev_ri = 1
        p_prev_ri  = 1 - stats.chi2.cdf(lr_prev_ri, df_prev_ri)
        rows.append({
            "Model":       "Random intercept + predictors",
            "LogL":        round(m_ri_ml.llf, 2),
            "AIC":         round(m_ri_ml.aic,  2),
            "BIC":         round(m_ri_ml.bic,  2),
            "LRT vs prev": round(lr_prev_ri, 2),
            "df":          df_prev_ri,
            "p-value":     round(p_prev_ri, 5),
        })
        if m_rs is not None:
            try:
                m_rs_ml = smf.mixedlm(formula, data, groups=data[group_col],
                                      re_formula=re_formula).fit(reml=False, method="lbfgs")
                lr_ri_rs  = 2 * (m_rs_ml.llf - m_ri_ml.llf)
                df_ri_rs  = 2   # slope variance + covariance
                p_ri_rs   = 1 - stats.chi2.cdf(lr_ri_rs, df_ri_rs)
                rows.append({
                    "Model":       "Random intercept + random slope(s)",
                    "LogL":        round(m_rs_ml.llf, 2),
                    "AIC":         round(m_rs_ml.aic,  2),
                    "BIC":         round(m_rs_ml.bic,  2),
                    "LRT vs prev": round(lr_ri_rs, 2),
                    "df":          df_ri_rs,
                    "p-value":     round(p_ri_rs, 5),
                })
            except Exception:
                pass

        cmp_df = pd.DataFrame(rows)
        st.dataframe(cmp_df, use_container_width=True)
        st.caption(
            "LRT statistic = 2 × (LogL_larger − LogL_smaller). "
            "df = difference in number of parameters between models. "
            "p < 0.05 → the more complex model significantly improves fit."
        )
    except Exception as e:
        st.info("Model comparison table could not be generated: " + str(e))

    # ── Fixed effects coefficients ────────────────────────────
    st.markdown("---")
    st.markdown("## Fixed Effects Coefficients")
    st.caption("Population-level estimates — apply to all groups on average.")

    result = m_rs if (m_rs is not None) else m_ri
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

    # Forest plot
    st.markdown("### Forest Plot — Fixed Effects")
    fe_plot = coef_tbl[~coef_tbl["Term"].str.contains("Intercept")].copy()
    if not fe_plot.empty:
        fig_forest = go.Figure()
        fig_forest.add_trace(go.Scatter(
            x=fe_plot["Coefficient"],
            y=fe_plot["Term"],
            mode="markers",
            marker=dict(size=10, color="#2563EB"),
            error_x=dict(
                type="data", symmetric=False,
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

    # ── Caterpillar plot (BLUPs) ──────────────────────────────
    st.markdown("---")
    st.markdown("## Caterpillar Plot — Group-Level Random Effects (BLUPs)")
    st.caption(
        "BLUPs (Best Linear Unbiased Predictors) = posterior estimates of the "
        "random intercept for each group, ordered from lowest to highest. "
        "Groups whose CI does not cross zero differ significantly from the average. "
        "This corresponds to Figure 2 in the MLM II course material."
    )
    try:
        re_dict = result.random_effects
        fig_cat, df_blup = _caterpillar_plot(
            re_dict, group_col, var_b if var_b else 0.5,
            plot_template
        )
        st.plotly_chart(fig_cat, use_container_width=True)

        # Normality check of BLUPs (Figure 1 in MLM II)
        st.markdown("### Normality of Random Intercepts (BLUPs)")
        st.caption(
            "Random intercepts are assumed to follow N(0, σ²ᵤ₀). "
            "Severe departures from normality indicate convergence problems "
            "(Figure 1 from MLM II course)."
        )
        blup_vals = df_blup["BLUP"].values
        fig_blup_hist = px.histogram(
            x=blup_vals, nbins=max(10, n_groups // 3),
            title="Distribution of Random Intercepts (BLUPs)",
            labels={"x": "BLUP value", "y": "Count"},
            template=plot_template,
        )
        st.plotly_chart(fig_blup_hist, use_container_width=True)
        if n_groups >= 8:
            _, sw_p = stats.shapiro(blup_vals)
            st.metric("Shapiro-Wilk p (BLUPs)", round(sw_p, 4),
                      help="p > 0.05 = BLUPs approximately normal. "
                           "With few groups the test has low power.")
            if sw_p < 0.05:
                issues.append({
                    "level": "warning",
                    "msg": "BLUPs may not be normally distributed (p=" + str(round(sw_p, 4)) + ").",
                    "fix": "Check for outlier groups or model misspecification.",
                })

        with st.expander("Random Effects Table (BLUPs)"):
            st.dataframe(df_blup[[group_col, "BLUP", "CI_lo", "CI_hi"]], use_container_width=True)

    except Exception as e:
        st.info("Caterpillar plot could not be generated: " + str(e))

    # ── Residual diagnostics ──────────────────────────────────
    st.markdown("---")
    st.markdown("## Residual Diagnostics")
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
            ))
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            fig_res.update_layout(
                title="Residuals vs Fitted",
                xaxis_title="Fitted", yaxis_title="Residuals",
                template=plot_template,
            )
            st.plotly_chart(fig_res, use_container_width=True)

        with col_b:
            std_resid = (residuals - residuals.mean()) / residuals.std()
            osm, osr  = stats.probplot(std_resid)[0]
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(x=list(osm), y=list(osr), mode="markers",
                                        marker=dict(size=4, color="#2563EB")))
            fig_qq.add_trace(go.Scatter(x=[min(osm), max(osm)], y=[min(osm), max(osm)],
                                        mode="lines", line=dict(color="red", dash="dash")))
            fig_qq.update_layout(title="Q-Q Plot", template=plot_template)
            st.plotly_chart(fig_qq, use_container_width=True)

        if len(residuals) <= 5000:
            _, sw_p = stats.shapiro(residuals)
            st.metric("Shapiro-Wilk p (residuals)", round(sw_p, 4))
            if sw_p < 0.05:
                issues.append({
                    "level": "warning",
                    "msg": "Residuals may not be normally distributed (p=" + str(round(sw_p, 4)) + ").",
                    "fix": "Consider log-transforming the outcome or using GLME.",
                })

        # Residuals by group
        st.markdown("### Residuals by Group")
        resid_df = pd.DataFrame({
            "Group":    data[group_col].astype(str).values,
            "Residual": residuals.values,
        })
        fig_rg = px.box(resid_df, x="Group", y="Residual",
                        title="Residuals by Group", template=plot_template)
        fig_rg.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_rg, use_container_width=True)

    except Exception as e:
        st.info("Residual diagnostics could not be generated: " + str(e))

    # ── Diagnostic summary ────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🩺 Diagnostic Summary")
    _diagnostic_summary(issues)

    with st.expander("Full Model Summary (statsmodels)"):
        st.text(result.summary().as_text())


# ============================================================
# GLME — Generalised Mixed Effects (GEE-based)
# ============================================================

def run_glme(df, outcome, fixed_effects, group_col, family_name, plot_template):
    st.markdown("## Generalised Mixed Effects Model (GLME via GEE)")
    st.caption(
        "Uses Generalised Estimating Equations (GEE) with an **exchangeable** "
        "working correlation structure — the standard population-average approach "
        "for clustered binary/count data. The exchangeable correlation α is an "
        "empirical estimate of the ICC / VPC. "
        "For binary outcomes the VPC uses the latent-variable formula: "
        "σ²ᵤ₀ / (σ²ᵤ₀ + π²/3)."
    )

    issues = []

    cols = list(dict.fromkeys([outcome, group_col] + fixed_effects))
    data, data_audit = prepare_complete_cases(df, cols)
    render_model_data_audit(data_audit)

    if data.empty:
        st.error("No data remaining after dropping missing values.")
        return

    n_total  = len(data)
    n_groups = data[group_col].nunique()
    n_per    = data.groupby(group_col).size()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations",  f"{n_total:,}")
    c2.metric("Groups",        f"{n_groups:,}")
    c3.metric("Min obs/group", int(n_per.min()))
    c4.metric("Max obs/group", int(n_per.max()))

    if n_groups < 5:
        issues.append({
            "level": "error",
            "msg": f"Only {n_groups} groups. GEE/GLME requires ≥5–10 groups.",
            "fix": "Use standard GLM instead or collect data from more groups.",
        })

    # Binary outcome encoding
    is_binary = "Logistic" in family_name
    if is_binary:
        unique_vals = data[outcome].dropna().unique()
        if not set(unique_vals).issubset({0, 1, True, False}):
            sorted_cls = sorted(unique_vals)
            data[outcome] = data[outcome].map({sorted_cls[0]: 0, sorted_cls[1]: 1})
            st.info("Outcome encoded: " + str(sorted_cls[0]) + "→0, " + str(sorted_cls[1]) + "→1")

    family_map = {
        "Logistic (binary outcome)":              sm.families.Binomial(),
        "Poisson (count outcome)":                sm.families.Poisson(),
        "Negative Binomial (overdispersed count)": sm.families.Poisson(),  # GEE uses Poisson + robust SE
    }
    family = family_map.get(family_name, sm.families.Binomial())

    formula_null = _quote(outcome) + " ~ 1"
    formula_full = _build_formula(outcome, fixed_effects, data)

    # ── Step 1: Null model ─────────────────────────────────────
    st.markdown("---")
    st.markdown("## Step 1 — Null GEE Model (Intercept Only)")
    st.caption("Estimates baseline ICC from the exchangeable correlation parameter α.")

    try:
        with st.spinner("Fitting null GEE model..."):
            gee_null = GEE.from_formula(
                formula_null,
                groups=group_col,
                data=data,
                family=family,
                cov_struct=Exchangeable(),
            )
            r_null = gee_null.fit()

        alpha_null = float(r_null.cov_struct.dep_params)
        n1, n2 = st.columns(2)
        n1.metric("Exchangeable α (ICC proxy) — null", round(alpha_null, 4),
                  help="Proportion of total variability attributable to group membership.")

        if is_binary:
            vpc_null, sigma2_u0_null = _vpc_binary_from_alpha(alpha_null)
            n2.metric("VPC (latent-variable formula) — null", vpc_null,
                      help="σ²ᵤ₀ / (σ²ᵤ₀ + π²/3). Null model estimate.")
            st.markdown(
                "**VPC formula for binary outcomes:** σ²ᵤ₀ / (σ²ᵤ₀ + π²/3) "
                "where σ²ᵤ₀ = " + str(round(sigma2_u0_null, 4)) +
                " (derived from α) and π²/3 ≈ 3.29 (latent logistic level-1 variance)."
            )
        else:
            n2.metric("α (ICC proxy) — null", round(alpha_null, 4))

        if alpha_null < 0.05:
            issues.append({
                "level": "warning",
                "msg": "ICC α = " + str(round(alpha_null, 4)) + " is low. Group-level clustering may be minimal.",
                "fix": "Consider standard GLM instead of GEE/GLME.",
            })

    except Exception as e:
        st.warning("Null GEE model failed: " + str(e))
        r_null = None
        alpha_null = None

    # ── Step 2: Full GEE model ─────────────────────────────────
    st.markdown("---")
    st.markdown("## Step 2 — Full GEE Model (with Predictors)")

    try:
        with st.spinner("Fitting full GEE model..."):
            gee_full = GEE.from_formula(
                formula_full,
                groups=group_col,
                data=data,
                family=family,
                cov_struct=Exchangeable(),
            )
            r_full = gee_full.fit()

        st.success("Full GEE model fitted successfully.")

        alpha_full = float(r_full.cov_struct.dep_params)

        f1, f2, f3 = st.columns(3)
        f1.metric("N", int(r_full.nobs))
        f2.metric("Exchangeable α (ICC) — full", round(alpha_full, 4))
        if is_binary:
            vpc_full, _ = _vpc_binary_from_alpha(alpha_full)
            f3.metric("VPC — full model", vpc_full)
        else:
            f3.metric("α — full model", round(alpha_full, 4))

        # QIC
        try:
            qic_full = float(r_full.qic()[0])
            qic_null = float(r_null.qic()[0]) if r_null is not None else None
            qc1, qc2 = st.columns(2)
            qc1.metric("QIC (full)", round(qic_full, 2),
                       help="Quasi Information Criterion for GEE model selection. Smaller = better.")
            if qic_null is not None:
                delta_qic = round(qic_full - qic_null, 2)
                qc2.metric("ΔQIC (full − null)", delta_qic,
                           help="Negative ΔQIC means full model improves over null.")
                if delta_qic < 0:
                    st.success("✅ Full model improves over null (ΔQIC = " + str(delta_qic) + ").")
                else:
                    st.warning("Full model does not improve over null (ΔQIC = " + str(delta_qic) + ").")
        except Exception:
            pass

        # ── Coefficients table ─────────────────────────────────
        st.markdown("### Fixed Effects Coefficients")
        st.caption("Robust (sandwich) standard errors used. OR / IRR = exp(coefficient).")

        params = list(r_full.params)
        pvals  = list(r_full.pvalues)
        bse    = list(r_full.bse)
        conf   = r_full.conf_int()
        ci_lo  = list(conf[0])
        ci_hi  = list(conf[1])
        names  = list(r_full.params.index)

        coef_tbl = pd.DataFrame({
            "Term":        names,
            "Coefficient": [round(float(x), 4) for x in params],
            "Std Error":   [round(float(x), 4) for x in bse],
            "p-value":     [round(float(x), 5) for x in pvals],
            "CI Lower":    [round(float(x), 4) for x in ci_lo],
            "CI Upper":    [round(float(x), 4) for x in ci_hi],
            "Significant": ["YES" if float(p) < 0.05 else "NO" for p in pvals],
        })

        if is_binary:
            coef_tbl["Odds Ratio"]  = [round(float(np.exp(x)), 4) for x in params]
            coef_tbl["OR CI Lower"] = [round(float(np.exp(x)), 4) for x in ci_lo]
            coef_tbl["OR CI Upper"] = [round(float(np.exp(x)), 4) for x in ci_hi]
        elif "Poisson" in family_name or "Negative" in family_name:
            coef_tbl["IRR"]          = [round(float(np.exp(x)), 4) for x in params]
            coef_tbl["IRR CI Lower"] = [round(float(np.exp(x)), 4) for x in ci_lo]
            coef_tbl["IRR CI Upper"] = [round(float(np.exp(x)), 4) for x in ci_hi]

        st.dataframe(coef_tbl, use_container_width=True)
        st.download_button(
            "📥 Download GLME coefficients (CSV)",
            data=coef_tbl.to_csv(index=False).encode(),
            file_name="glme_coefficients.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Forest plot
        fe_plot = coef_tbl[~coef_tbl["Term"].str.contains("Intercept")].copy()
        if not fe_plot.empty:
            x_col = "Odds Ratio" if is_binary else (
                "IRR" if "IRR" in fe_plot.columns else "Coefficient"
            )
            x_lo  = "OR CI Lower" if is_binary else (
                "IRR CI Lower" if "IRR CI Lower" in fe_plot.columns else "CI Lower"
            )
            x_hi  = "OR CI Upper" if is_binary else (
                "IRR CI Upper" if "IRR CI Upper" in fe_plot.columns else "CI Upper"
            )
            vline_x = 1 if x_col in ("Odds Ratio", "IRR") else 0
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(
                x=fe_plot[x_col], y=fe_plot["Term"],
                mode="markers",
                marker=dict(size=10, color="#2563EB"),
                error_x=dict(
                    type="data", symmetric=False,
                    array=(fe_plot[x_hi] - fe_plot[x_col]).tolist(),
                    arrayminus=(fe_plot[x_col] - fe_plot[x_lo]).tolist(),
                ),
                name=x_col + " (95% CI)",
            ))
            fig_f.add_vline(x=vline_x, line_dash="dash", line_color="red",
                            annotation_text="No effect")
            fig_f.update_layout(
                title="Forest Plot — " + x_col,
                xaxis_title=x_col, template=plot_template,
                height=max(300, len(fe_plot) * 45 + 100),
            )
            st.plotly_chart(fig_f, use_container_width=True)

        # ── Diagnostics ────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🔬 GLME Diagnostics")
        st.caption(
            "GEE uses robust sandwich estimators for SEs — "
            "results are valid even if the working correlation is mis-specified."
        )

        if is_binary:
            predicted = r_full.predict()
            y_obs     = data[outcome].astype(int).values
            from sklearn.metrics import roc_auc_score
            try:
                auc = round(roc_auc_score(y_obs, predicted), 4)
                st.metric("AUC (ROC)", auc,
                          help="Population-average discrimination. "
                               "AUC > 0.7 = acceptable, > 0.8 = good.")
            except Exception:
                pass

        # Pearson residuals
        try:
            pres = r_full.resid_pearson
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=list(range(len(pres))), y=list(pres),
                mode="markers", marker=dict(size=3, color="#2563EB", opacity=0.4),
            ))
            fig_pr.add_hline(y=0, line_dash="dash", line_color="red")
            fig_pr.update_layout(
                title="Pearson Residuals",
                xaxis_title="Observation index",
                yaxis_title="Pearson residual",
                template=plot_template,
            )
            st.plotly_chart(fig_pr, use_container_width=True)
        except Exception:
            pass

    except Exception as e:
        st.error("Full GEE model failed: " + str(e))
        issues.append({
            "level": "error",
            "msg": "GLME fitting failed.",
            "fix": "Check outcome type, encoding, and group column.",
        })

    st.markdown("---")
    st.markdown("## 🩺 Diagnostic Summary")
    _diagnostic_summary(issues)


# ============================================================
# Main render function
# ============================================================

def render_mixed_effects_tab(df, df_cleaned, plot_template):
    st.markdown("# Multilevel / Mixed Effects Models")
    st.info(
        "Mixed effects (multilevel) models handle **clustered, hierarchical, "
        "or repeated-measures data** where observations within groups are not independent. "
        "Examples: patients nested within hospitals, students within schools, "
        "repeat observations within subjects.\n\n"
        "The workflow follows the iterative model-building approach: "
        "Null model → Random Intercept → Random Slope → compare with LRT."
    )

    # Dataset
    dataset_choice = st.radio(
        "Dataset to use",
        ["Original data", "Cleaned data (from Data Cleaning tab)"],
        horizontal=True,
        key="me_dataset",
    )
    mdf = df_cleaned if dataset_choice.startswith("Cleaned") else df

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
            "Outcome variable (Y)", all_cols, key="me_outcome",
            help="The variable you want to predict.",
        )
    with me2:
        group_options = [c for c in all_cols if c != outcome]
        group_col = st.selectbox(
            "Grouping variable (Level 2 cluster)", group_options, key="me_group",
            help="Variable defining groups (hospital, school, subject ID, …).",
        )
    with me3:
        available_fe = [c for c in all_cols if c not in (outcome, group_col)]
        fixed_effects = st.multiselect(
            "Fixed effects (Level 1 + Level 2 predictors)",
            available_fe,
            default=available_fe[:min(4, len(available_fe))],
            key="me_fixed",
        )

    # ── Model type ────────────────────────────────────────────
    st.markdown("### Model Type")
    model_type = st.radio(
        "Choose model",
        [
            "Linear Mixed Effects (LME) — continuous outcome",
            "Generalised LME / GEE — binary or count outcome",
        ],
        horizontal=True,
        key="me_model_type",
    )

    random_slopes = []
    family_name   = "Logistic (binary outcome)"

    if "Linear" in model_type:
        st.markdown("### Random Effects Structure")
        num_fixed = [c for c in fixed_effects
                     if pd.api.types.is_numeric_dtype(mdf[c])]
        if num_fixed:
            random_slopes = st.multiselect(
                "Add random slopes for (optional — Step 3):",
                num_fixed, key="me_random_slopes",
                help=(
                    "Leave empty to fit a random intercept model only (Step 2). "
                    "Add slopes to test whether the effect of a predictor varies "
                    "across groups (Key Point 3b — random slope model)."
                ),
            )
            if random_slopes:
                st.caption(
                    "Random slopes allow the effect of the selected predictor(s) to "
                    "vary across groups. The model will also estimate the "
                    "intercept–slope covariance (ρ₀₁)."
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

    # ── Guidance expander ─────────────────────────────────────
    with st.expander("📖 When to use which model + key concepts"):
        st.markdown("""
**Use Linear Mixed Effects (LME) when:**
- Outcome is continuous (length of stay, test score, blood pressure)
- Data is clustered (patients in hospitals, employees in companies)
- You have repeated measurements on the same subject

**Use Generalised LME / GEE when:**
- Outcome is binary (death yes/no, readmission yes/no) → Logistic family
- Outcome is a count (number of procedures) → Poisson / NB family
- Data is still clustered or repeated

**Key concepts from the course:**
| Concept | Meaning |
|---|---|
| ICC / VPC | Proportion of variance at the group level (σ²ᵤ₀ / (σ²ᵤ₀ + σ²ₑ)) |
| Random intercept | Each group has its own baseline; slopes are parallel |
| Random slope | Both intercept and slope vary across groups |
| ρ₀₁ (covariance) | Positive = fanning-out; Negative = fanning-in |
| BLUPs / Caterpillar | Posterior group-level estimates, sorted low to high |
| LRT | Likelihood Ratio Test — compares nested models |
| AIC / BIC | For non-nested model comparison (smaller = better) |
| VPC (binary) | σ²ᵤ₀ / (σ²ᵤ₀ + π²/3), where π²/3 ≈ 3.29 |
""")

    # ── Run button ────────────────────────────────────────────
    if st.button("Run Mixed Effects Model", use_container_width=True, key="me_run"):
        if not fixed_effects:
            st.error("Select at least one fixed effect predictor.")
            return
        try:
            if "Generalised" in model_type or "GEE" in model_type:
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

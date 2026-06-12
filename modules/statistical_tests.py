# ============================================================
# modules/statistical_tests.py
# Statistical Tests with automatic interpretation
# T-test, ANOVA, Chi-square, Normality, Correlation
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
# Helpers
# ============================================================

def _interpret_pval(pval, alpha=0.05):
    if pval < 0.001:
        return "p < 0.001 - Very strong evidence against H0"
    elif pval < 0.01:
        return "p < 0.01 - Strong evidence against H0"
    elif pval < 0.05:
        return "p < 0.05 - Significant at 5% level"
    elif pval < 0.10:
        return "p < 0.10 - Marginal significance"
    else:
        return "p >= 0.05 - No significant evidence against H0"


def _effect_size_label(d):
    d = abs(d)
    if d < 0.2:   return "Negligible"
    elif d < 0.5: return "Small"
    elif d < 0.8: return "Medium"
    else:         return "Large"


def _show_result(stat_name, stat_val, pval, significant, interpretation, effect=None):
    col1, col2, col3 = st.columns(3)
    col1.metric(stat_name, round(stat_val, 4))
    col2.metric("p-value", round(pval, 5))
    col3.metric("Result", "Significant" if significant else "Not significant")
    st.caption(_interpret_pval(pval))
    if effect:
        st.caption("Effect size: " + effect)
    if significant:
        st.success("YES - " + interpretation)
    else:
        st.info("NO - " + interpretation)


# ============================================================
# 1. Normality Tests
# ============================================================

def render_normality_tests(df, numeric_cols, plot_template):
    st.markdown("### Normality Tests")
    st.caption(
        "Tests whether a variable follows a normal distribution. "
        "H0: data is normally distributed. p < 0.05 = reject normality."
    )

    col = st.selectbox("Select variable", numeric_cols, key="norm_col")
    series = df[col].dropna()
    n = len(series)

    if n < 3:
        st.warning("Need at least 3 observations.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N", n)
    c2.metric("Mean",     round(series.mean(), 3))
    c3.metric("Skewness", round(series.skew(), 3))
    c4.metric("Kurtosis", round(series.kurtosis(), 3))

    col_a, col_b = st.columns(2)

    with col_a:
        fig_hist = px.histogram(
            series, nbins=30, marginal="box",
            title="Distribution of " + col,
            template=plot_template,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        osm, osr = stats.probplot(series)[0]
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
            title="Q-Q Plot",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            template=plot_template,
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    results = []

    # Shapiro-Wilk (best for n <= 5000)
    if n <= 5000:
        sw_stat, sw_p = stats.shapiro(series)
        results.append({
            "Test": "Shapiro-Wilk",
            "Statistic": round(sw_stat, 4),
            "p-value": round(sw_p, 5),
            "Normal": "YES" if sw_p >= 0.05 else "NO",
            "Note": "Best for n <= 5000",
        })

    # Kolmogorov-Smirnov
    ks_stat, ks_p = stats.kstest(
        (series - series.mean()) / series.std(), "norm"
    )
    results.append({
        "Test": "Kolmogorov-Smirnov",
        "Statistic": round(ks_stat, 4),
        "p-value": round(ks_p, 5),
        "Normal": "YES" if ks_p >= 0.05 else "NO",
        "Note": "Good for large samples",
    })

    # D'Agostino-Pearson
    if n >= 8:
        dp_stat, dp_p = stats.normaltest(series)
        results.append({
            "Test": "D'Agostino-Pearson",
            "Statistic": round(dp_stat, 4),
            "p-value": round(dp_p, 5),
            "Normal": "YES" if dp_p >= 0.05 else "NO",
            "Note": "Tests skewness + kurtosis",
        })

    st.dataframe(pd.DataFrame(results), use_container_width=True)

    # Skewness interpretation
    skew = series.skew()
    if abs(skew) > 1:
        st.warning(
            "Skewness = " + str(round(skew, 3)) +
            " - Highly skewed distribution. "
            "Consider log-transform before parametric tests."
        )
    elif abs(skew) > 0.5:
        st.info("Skewness = " + str(round(skew, 3)) + " - Moderately skewed.")
    else:
        st.success("Skewness = " + str(round(skew, 3)) + " - Approximately symmetric.")


# ============================================================
# 2. T-tests
# ============================================================

def render_ttest(df, numeric_cols, categorical_cols, plot_template):
    st.markdown("### T-Test")

    ttest_type = st.radio(
        "Test type",
        ["One-sample T-test", "Independent T-test (2 groups)", "Paired T-test"],
        horizontal=True,
        key="tt_type",
    )

    if ttest_type == "One-sample T-test":
        st.caption("Tests if the mean of a variable equals a hypothesized value.")
        t1, t2 = st.columns(2)
        col  = t1.selectbox("Variable", numeric_cols, key="tt1_col")
        mu   = t2.number_input("Hypothesized mean (H0)", value=0.0, key="tt1_mu")
        series = df[col].dropna()

        if st.button("Run One-Sample T-test", key="tt1_run"):
            t_stat, p_val = stats.ttest_1samp(series, mu)
            ci = stats.t.interval(0.95, len(series)-1,
                                  loc=series.mean(),
                                  scale=stats.sem(series))
            d = (series.mean() - mu) / series.std()

            _show_result(
                "T-statistic", t_stat, p_val, p_val < 0.05,
                "The mean of " + col + " differs significantly from " + str(mu),
                effect=_effect_size_label(d) + " (Cohen's d = " + str(round(d,3)) + ")",
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Sample Mean",   round(series.mean(), 4))
            c2.metric("95% CI Lower",  round(ci[0], 4))
            c3.metric("95% CI Upper",  round(ci[1], 4))

    elif ttest_type == "Independent T-test (2 groups)":
        st.caption(
            "Tests if two independent groups have different means. "
            "H0: mean(group1) = mean(group2)."
        )
        t1, t2, t3 = st.columns(3)
        col      = t1.selectbox("Numeric variable", numeric_cols, key="tt2_col")
        grp_col  = t2.selectbox("Grouping variable", categorical_cols + numeric_cols, key="tt2_grp")
        equal_var = t3.checkbox("Assume equal variances", value=False, key="tt2_eqvar")

        if grp_col and col:
            groups = df[grp_col].dropna().unique()
            if len(groups) == 2:
                g1 = df[df[grp_col] == groups[0]][col].dropna()
                g2 = df[df[grp_col] == groups[1]][col].dropna()

                # Descriptive stats
                desc = pd.DataFrame({
                    "Group":  [str(groups[0]), str(groups[1])],
                    "N":      [len(g1), len(g2)],
                    "Mean":   [round(g1.mean(),3), round(g2.mean(),3)],
                    "Std":    [round(g1.std(),3),  round(g2.std(),3)],
                    "Median": [round(g1.median(),3), round(g2.median(),3)],
                })
                st.dataframe(desc, use_container_width=True)

                # Box plot
                plot_df = pd.DataFrame({
                    col: pd.concat([g1, g2]),
                    grp_col: [str(groups[0])]*len(g1) + [str(groups[1])]*len(g2),
                })
                fig = px.box(
                    plot_df, x=grp_col, y=col,
                    color=grp_col,
                    title=col + " by " + grp_col,
                    template=plot_template,
                )
                st.plotly_chart(fig, use_container_width=True)

                if st.button("Run Independent T-test", key="tt2_run"):
                    # Levene test first
                    lev_stat, lev_p = stats.levene(g1, g2)
                    st.caption(
                        "Levene test (equal variances): F=" + str(round(lev_stat,3)) +
                        ", p=" + str(round(lev_p,4)) +
                        (" - Variances are equal" if lev_p >= 0.05 else " - Variances differ (use Welch)")
                    )

                    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=equal_var)
                    pooled_std = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
                    d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0

                    _show_result(
                        "T-statistic", t_stat, p_val, p_val < 0.05,
                        "The means of " + str(groups[0]) + " and " + str(groups[1]) +
                        " differ significantly for " + col,
                        effect=_effect_size_label(d) + " (Cohen's d = " + str(round(d,3)) + ")",
                    )

                    # Non-parametric alternative
                    mw_stat, mw_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                    st.caption(
                        "Mann-Whitney U (non-parametric): U=" + str(round(mw_stat,2)) +
                        ", p=" + str(round(mw_p,4))
                    )
            else:
                st.warning(
                    "Grouping variable has " + str(len(groups)) + " unique values. "
                    "Need exactly 2 for T-test. Use ANOVA for more groups."
                )

    elif ttest_type == "Paired T-test":
        st.caption(
            "Tests if two related measurements differ. "
            "H0: mean difference = 0."
        )
        p1, p2 = st.columns(2)
        col1 = p1.selectbox("Variable 1 (before)", numeric_cols, key="ttp1")
        col2 = p2.selectbox("Variable 2 (after)",  numeric_cols, key="ttp2")

        if col1 != col2:
            paired = df[[col1, col2]].dropna()
            diff   = paired[col1] - paired[col2]

            d1, d2, d3 = st.columns(3)
            d1.metric("Mean difference", round(diff.mean(), 4))
            d2.metric("Std of diff",     round(diff.std(), 4))
            d3.metric("N pairs",         len(paired))

            fig_diff = px.histogram(
                diff, nbins=20,
                title="Distribution of Differences (" + col1 + " - " + col2 + ")",
                template=plot_template,
            )
            st.plotly_chart(fig_diff, use_container_width=True)

            if st.button("Run Paired T-test", key="ttp_run"):
                t_stat, p_val = stats.ttest_rel(paired[col1], paired[col2])
                d = diff.mean() / diff.std() if diff.std() > 0 else 0
                _show_result(
                    "T-statistic", t_stat, p_val, p_val < 0.05,
                    col1 + " and " + col2 + " differ significantly",
                    effect=_effect_size_label(d) + " (Cohen's d = " + str(round(d,3)) + ")",
                )


# ============================================================
# 3. ANOVA
# ============================================================

def render_anova(df, numeric_cols, categorical_cols, plot_template):
    st.markdown("### ANOVA — Analysis of Variance")
    st.caption(
        "Tests if means differ across 3 or more groups. "
        "H0: all group means are equal."
    )

    a1, a2 = st.columns(2)
    col     = a1.selectbox("Numeric variable", numeric_cols, key="anov_col")
    grp_col = a2.selectbox("Grouping variable", categorical_cols + numeric_cols, key="anov_grp")

    groups_list = df[grp_col].dropna().unique()
    n_groups = len(groups_list)

    if n_groups < 2:
        st.warning("Need at least 2 groups.")
        return
    if n_groups > 20:
        st.warning("Too many groups (" + str(n_groups) + "). Select a variable with fewer categories.")
        return

    # Descriptive stats per group
    desc_rows = []
    group_data = []
    for grp in sorted(groups_list):
        s = df[df[grp_col] == grp][col].dropna()
        group_data.append(s)
        desc_rows.append({
            "Group": str(grp),
            "N":     len(s),
            "Mean":  round(s.mean(), 3),
            "Std":   round(s.std(), 3),
            "Median":round(s.median(), 3),
        })
    st.dataframe(pd.DataFrame(desc_rows), use_container_width=True)

    # Box plot
    plot_df = df[[col, grp_col]].dropna().copy()
    plot_df[grp_col] = plot_df[grp_col].astype(str)
    fig = px.box(
        plot_df, x=grp_col, y=col,
        color=grp_col,
        title=col + " by " + grp_col,
        template=plot_template,
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Run ANOVA", key="anov_run"):
        f_stat, p_val = stats.f_oneway(*group_data)

        # Eta-squared effect size
        grand_mean = np.concatenate([g.values for g in group_data]).mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in group_data)
        ss_total   = sum(((g - grand_mean)**2).sum() for g in group_data)
        eta_sq     = ss_between / ss_total if ss_total > 0 else 0

        _show_result(
            "F-statistic", f_stat, p_val, p_val < 0.05,
            "At least one group mean differs significantly",
            effect="Eta-squared = " + str(round(eta_sq, 4)) +
                   " (" + _effect_size_label(np.sqrt(eta_sq)) + ")",
        )

        # Post-hoc Tukey HSD if significant
        if p_val < 0.05 and n_groups <= 10:
            st.markdown("#### Post-hoc: Tukey HSD")
            st.caption("Pairwise comparisons after significant ANOVA.")
            try:
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                all_data = pd.concat([
                    pd.DataFrame({"value": g, "group": str(grp)})
                    for g, grp in zip(group_data, sorted(groups_list))
                ])
                tukey = pairwise_tukeyhsd(all_data["value"], all_data["group"])
                tukey_df = pd.DataFrame(
                    data=tukey._results_table.data[1:],
                    columns=tukey._results_table.data[0],
                )
                st.dataframe(tukey_df, use_container_width=True)
            except Exception as e:
                st.info("Tukey HSD could not be computed: " + str(e))

        # Non-parametric alternative
        kw_stat, kw_p = stats.kruskal(*group_data)
        st.caption(
            "Kruskal-Wallis (non-parametric alternative): "
            "H=" + str(round(kw_stat,3)) + ", p=" + str(round(kw_p,4))
        )


# ============================================================
# 4. Chi-Square Test
# ============================================================

def render_chi_square(df, categorical_cols, numeric_cols, plot_template):
    st.markdown("### Chi-Square Test of Independence")
    st.caption(
        "Tests if two categorical variables are associated. "
        "H0: the variables are independent."
    )

    all_cols = categorical_cols + numeric_cols
    ch1, ch2 = st.columns(2)
    col1 = ch1.selectbox("Variable 1", all_cols, key="chi1")
    col2 = ch2.selectbox("Variable 2", all_cols, key="chi2")

    if col1 == col2:
        st.warning("Select two different variables.")
        return

    data = df[[col1, col2]].dropna()
    ct = pd.crosstab(data[col1], data[col2])

    if ct.shape[0] < 2 or ct.shape[1] < 2:
        st.warning("Each variable needs at least 2 categories.")
        return

    if ct.shape[0] > 15 or ct.shape[1] > 15:
        st.warning("Too many categories for chi-square. Select variables with fewer categories.")
        return

    st.markdown("#### Contingency Table")
    st.dataframe(ct, use_container_width=True)

    # Heatmap
    fig_ct = px.imshow(
        ct, text_auto=True, aspect="auto",
        color_continuous_scale="Blues",
        title="Contingency Table Heatmap",
    )
    st.plotly_chart(fig_ct, use_container_width=True)

    if st.button("Run Chi-Square Test", key="chi_run"):
        chi2, p_val, dof, expected = stats.chi2_contingency(ct)

        # Check expected frequencies
        low_exp = (expected < 5).sum()
        if low_exp > 0:
            st.warning(
                str(low_exp) + " cells have expected frequency < 5. "
                "Consider Fisher's Exact Test or merging categories."
            )

        _show_result(
            "Chi-square", chi2, p_val, p_val < 0.05,
            col1 + " and " + col2 + " are significantly associated",
        )
        st.metric("Degrees of freedom", dof)

        # Cramer's V effect size
        n = ct.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
        st.metric("Cramer's V (effect size)", round(cramers_v, 4),
                  help="0.1=small, 0.3=medium, 0.5=large")

        # Fisher's exact for 2x2
        if ct.shape == (2, 2):
            odds_ratio, fisher_p = stats.fisher_exact(ct)
            st.caption(
                "Fisher's Exact Test: OR=" + str(round(odds_ratio,3)) +
                ", p=" + str(round(fisher_p,4))
            )

        # Expected frequencies table
        with st.expander("Expected Frequencies"):
            exp_df = pd.DataFrame(
                expected.round(2),
                index=ct.index,
                columns=ct.columns,
            )
            st.dataframe(exp_df, use_container_width=True)


# ============================================================
# 5. Correlation Test
# ============================================================

def render_correlation_test(df, numeric_cols, plot_template):
    st.markdown("### Correlation Test")
    st.caption(
        "Tests the statistical significance of correlation between two variables."
    )

    cr1, cr2, cr3 = st.columns(3)
    col1   = cr1.selectbox("Variable 1", numeric_cols, key="corr1")
    col2   = cr2.selectbox("Variable 2", numeric_cols, key="corr2")
    method = cr3.selectbox("Method", ["Pearson", "Spearman", "Kendall"], key="corr_method")

    if col1 == col2:
        st.warning("Select two different variables.")
        return

    data = df[[col1, col2]].dropna()

    # Scatter plot
    fig_sc = px.scatter(
        data, x=col1, y=col2,
        trendline="ols",
        title=col1 + " vs " + col2,
        template=plot_template,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    if st.button("Run Correlation Test", key="corr_run"):
        if method == "Pearson":
            r, p_val = stats.pearsonr(data[col1], data[col2])
            stat_name = "Pearson r"
        elif method == "Spearman":
            r, p_val = stats.spearmanr(data[col1], data[col2])
            stat_name = "Spearman rho"
        else:
            r, p_val = stats.kendalltau(data[col1], data[col2])
            stat_name = "Kendall tau"

        _show_result(
            stat_name, r, p_val, p_val < 0.05,
            col1 + " and " + col2 + " are significantly correlated",
            effect=_effect_size_label(r) + " correlation",
        )

        # Confidence interval for Pearson
        if method == "Pearson":
            n = len(data)
            z = np.arctanh(r)
            se = 1 / np.sqrt(n - 3)
            z_ci = stats.norm.ppf(0.975)
            ci_lo = np.tanh(z - z_ci * se)
            ci_hi = np.tanh(z + z_ci * se)
            st.caption(
                "95% CI for r: [" + str(round(ci_lo,4)) + ", " + str(round(ci_hi,4)) + "]"
            )


# ============================================================
# 6. Levene Test (equal variances)
# ============================================================

def render_levene(df, numeric_cols, categorical_cols, plot_template):
    st.markdown("### Levene Test — Equality of Variances")
    st.caption(
        "Tests if two or more groups have equal variance. "
        "H0: all group variances are equal. "
        "Used before T-test and ANOVA."
    )

    lv1, lv2 = st.columns(2)
    col     = lv1.selectbox("Numeric variable", numeric_cols, key="lev_col")
    grp_col = lv2.selectbox("Grouping variable", categorical_cols + numeric_cols, key="lev_grp")

    groups_list = df[grp_col].dropna().unique()
    group_data  = [df[df[grp_col] == g][col].dropna() for g in groups_list]
    group_data  = [g for g in group_data if len(g) > 1]

    if len(group_data) < 2:
        st.warning("Need at least 2 groups with more than 1 observation.")
        return

    # Variance per group
    var_df = pd.DataFrame({
        "Group":    [str(g) for g in groups_list[:len(group_data)]],
        "Variance": [round(g.var(), 4) for g in group_data],
        "Std Dev":  [round(g.std(), 4) for g in group_data],
        "N":        [len(g) for g in group_data],
    })
    st.dataframe(var_df, use_container_width=True)

    if st.button("Run Levene Test", key="lev_run"):
        lev_stat, lev_p = stats.levene(*group_data)
        _show_result(
            "Levene F", lev_stat, lev_p, lev_p < 0.05,
            "Group variances are significantly different - use Welch T-test",
        )
        if lev_p >= 0.05:
            st.success("Equal variances assumption holds - standard T-test/ANOVA is appropriate.")
        else:
            st.warning(
                "Variances differ significantly. "
                "Use Welch T-test (unequal variances) or non-parametric tests."
            )


# ============================================================
# Main render function
# ============================================================

def render_statistical_tests(df, plot_template):
    st.markdown("## Statistical Tests")
    st.info(
        "Select a test type. Each test includes automatic interpretation, "
        "effect size calculation, and non-parametric alternatives where applicable."
    )

    numeric_cols     = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    if len(numeric_cols) == 0:
        st.warning("No numeric columns found.")
        return

    test_type = st.selectbox(
        "Select test",
        [
            "Normality Tests (Shapiro-Wilk, KS, D'Agostino)",
            "T-Test (One-sample, Independent, Paired)",
            "ANOVA (+ Post-hoc Tukey HSD)",
            "Chi-Square Test of Independence",
            "Correlation Test (Pearson, Spearman, Kendall)",
            "Levene Test (Equal Variances)",
        ],
        key="stat_test_type",
    )

    st.markdown("---")

    if test_type.startswith("Normality"):
        render_normality_tests(df, numeric_cols, plot_template)

    elif test_type.startswith("T-Test"):
        if not categorical_cols:
            st.warning("No categorical columns for grouping. Add a grouping variable.")
        render_ttest(df, numeric_cols, categorical_cols, plot_template)

    elif test_type.startswith("ANOVA"):
        if not categorical_cols:
            st.warning("No categorical columns for grouping.")
            return
        render_anova(df, numeric_cols, categorical_cols, plot_template)

    elif test_type.startswith("Chi-Square"):
        if len(categorical_cols) < 1:
            st.warning("Need at least one categorical variable.")
            return
        render_chi_square(df, categorical_cols, numeric_cols, plot_template)

    elif test_type.startswith("Correlation"):
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric variables.")
            return
        render_correlation_test(df, numeric_cols, plot_template)

    elif test_type.startswith("Levene"):
        if not categorical_cols:
            st.warning("No categorical columns for grouping.")
            return
        render_levene(df, numeric_cols, categorical_cols, plot_template)
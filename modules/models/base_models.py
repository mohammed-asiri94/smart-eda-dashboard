# ============================================================
# modules/models/base_models.py
# Linear, Logistic, Poisson, Negative Binomial
# Each with full diagnostics + fix suggestions
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, linear_harvey_collier
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import OLSInfluence

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from scipy import stats


# ============================================================
# Shared helpers
# ============================================================

def quote_col(col):
    return f'Q("{str(col).replace(chr(34), chr(92)+chr(34))}")'


def build_formula(target, predictors, df):
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    parts = [
        f"C({quote_col(c)})" if c in cat_cols else quote_col(c)
        for c in predictors
    ]
    return f"{quote_col(target)} ~ {' + '.join(parts)}"


def prepare_data(df, target, predictors):
    cols = [target] + predictors
    return df[cols].dropna().copy()


def coef_table(result, model_type):
    params = result.params
    conf = result.conf_int()
    tbl = pd.DataFrame({
        "Term": params.index,
        "Coefficient": params.values.round(4),
        "Std Error": result.bse.values.round(4),
        "P-value": result.pvalues.values.round(5),
        "CI Lower": conf[0].values.round(4),
        "CI Upper": conf[1].values.round(4),
    })
    if model_type == "Binary Logistic Regression":
        tbl["Odds Ratio"] = np.exp(params.values).round(4)
        tbl["OR CI Lower"] = np.exp(conf[0].values).round(4)
        tbl["OR CI Upper"] = np.exp(conf[1].values).round(4)
    if model_type in ("Poisson Regression", "Negative Binomial Regression"):
        tbl["IRR"] = np.exp(params.values).round(4)
        tbl["IRR CI Lower"] = np.exp(conf[0].values).round(4)
        tbl["IRR CI Upper"] = np.exp(conf[1].values).round(4)
    return tbl


def download_buttons(result, tbl, model_type):
    st.download_button(
        "📥 Download coefficients (CSV)",
        data=tbl.to_csv(index=False).encode(),
        file_name="coefficients.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        "📥 Download full model summary (TXT)",
        data=result.summary().as_text().encode(),
        file_name="model_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )


def show_diagnostics_header(issues):
    """Show traffic-light summary of diagnostic issues."""
    if not issues:
        st.success("✅ All diagnostic checks passed — no major issues detected.")
        return
    for issue in issues:
        level = issue.get("level", "warning")
        msg = issue["msg"]
        fix = issue.get("fix", "")
        if level == "error":
            st.error(f"🔴 **{msg}**\n\n💡 *Suggested fix:* {fix}")
        elif level == "warning":
            st.warning(f"🟡 **{msg}**\n\n💡 *Suggested fix:* {fix}")
        else:
            st.info(f"🔵 {msg}")


# ============================================================
# Linear Regression
# ============================================================

def run_linear_regression(df, target, predictors, plot_template):
    model_df = prepare_data(df, target, predictors)
    if model_df.empty:
        st.error("No data remaining after dropping missing values.")
        return

    formula = build_formula(target, predictors, model_df)
    result = smf.ols(formula=formula, data=model_df).fit()

    st.success("✅ Model fitted successfully.")
    st.code(formula, language="text")

    # ── Fit metrics ─────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N", int(result.nobs))
    c2.metric("R²", round(result.rsquared, 4))
    c3.metric("Adj. R²", round(result.rsquared_adj, 4))
    c4.metric("AIC", round(result.aic, 2))

    # ── Coefficients ─────────────────────────────────────────
    st.markdown("### Coefficients")
    tbl = coef_table(result, "Linear Regression")
    st.dataframe(tbl, use_container_width=True)
    download_buttons(result, tbl, "Linear Regression")

    # ── Predictions ─────────────────────────────────────────
    preds = result.predict(model_df)
    residuals = model_df[target] - preds

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.scatter(
            x=model_df[target], y=preds,
            labels={"x": "Actual", "y": "Predicted"},
            title="Actual vs Predicted",
            template=plot_template,
        )
        fig.add_shape(
            type="line",
            x0=model_df[target].min(), y0=model_df[target].min(),
            x1=model_df[target].max(), y1=model_df[target].max(),
            line=dict(dash="dash", color="red"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.scatter(
            x=preds, y=residuals,
            labels={"x": "Fitted Values", "y": "Residuals"},
            title="Residuals vs Fitted",
            template=plot_template,
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Q-Q Plot ─────────────────────────────────────────────
    st.markdown("### Q-Q Plot of Residuals")
    std_resid = (residuals - residuals.mean()) / residuals.std()
    (osm, osr), _ = stats.probplot(std_resid)
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(x=list(osm), y=list(osr), mode="markers", name="Residuals"))
    fig_qq.add_trace(go.Scatter(
        x=[min(osm), max(osm)], y=[min(osm), max(osm)],
        mode="lines", line=dict(color="red", dash="dash"), name="Normal line",
    ))
    fig_qq.update_layout(
        title="Normal Q-Q Plot", xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles", template=plot_template,
    )
    st.plotly_chart(fig_qq, use_container_width=True)

    # ── Diagnostics ──────────────────────────────────────────
    st.markdown("### 🔬 Diagnostics")
    issues = []

    # VIF
    try:
        num_predictors = model_df[predictors].select_dtypes(include=np.number).columns.tolist()
        if len(num_predictors) >= 2:
            X_vif = sm.add_constant(model_df[num_predictors].dropna())
            vif_data = pd.DataFrame({
                "Variable": X_vif.columns,
                "VIF": [
                    variance_inflation_factor(X_vif.values, i)
                    for i in range(X_vif.shape[1])
                ],
            })
            vif_data = vif_data[vif_data["Variable"] != "const"].round(3)
            st.markdown("#### Variance Inflation Factor (VIF)")
            st.dataframe(vif_data, use_container_width=True)
            high_vif = vif_data[vif_data["VIF"] > 10]
            if not high_vif.empty:
                issues.append({
                    "level": "error",
                    "msg": f"Multicollinearity detected — VIF > 10 for: {', '.join(high_vif['Variable'].tolist())}",
                    "fix": "Remove one of the correlated predictors, or switch to Ridge/Lasso regression.",
                })
            elif (vif_data["VIF"] > 5).any():
                issues.append({
                    "level": "warning",
                    "msg": "Moderate multicollinearity — VIF between 5 and 10 detected.",
                    "fix": "Consider removing or combining correlated predictors.",
                })
    except Exception:
        pass

    # Breusch-Pagan (Heteroscedasticity)
    try:
        bp_lm, bp_pval, _, _ = het_breuschpagan(result.resid, result.model.exog)
        st.markdown(
            f"**Breusch-Pagan test** — LM Statistic: `{bp_lm:.4f}`, p-value: `{bp_pval:.4f}`"
        )
        if bp_pval < 0.05:
            issues.append({
                "level": "warning",
                "msg": f"Heteroscedasticity detected (Breusch-Pagan p = {bp_pval:.4f})",
                "fix": "Use Robust Standard Errors (HC3), apply log-transform to the outcome, or use WLS.",
            })
    except Exception:
        pass

    # Durbin-Watson (Autocorrelation)
    try:
        dw = durbin_watson(result.resid)
        st.markdown(f"**Durbin-Watson statistic:** `{dw:.4f}` (ideal ≈ 2.0)")
        if dw < 1.5 or dw > 2.5:
            issues.append({
                "level": "warning",
                "msg": f"Possible autocorrelation in residuals (Durbin-Watson = {dw:.4f})",
                "fix": "If data is time-ordered, consider adding a lag variable or using GLS/ARIMA.",
            })
    except Exception:
        pass

    # Cook's Distance
    try:
        influence = OLSInfluence(result)
        cooks_d = influence.cooks_distance[0]
        threshold = 4 / len(model_df)
        n_influential = int((cooks_d > threshold).sum())
        st.markdown(
            f"**Cook's Distance** — {n_influential} influential observations "
            f"(threshold = {threshold:.4f})"
        )
        if n_influential > 0:
            cooks_df = pd.DataFrame({
                "Observation": np.where(cooks_d > threshold)[0],
                "Cook's D": cooks_d[cooks_d > threshold].round(5),
            })
            st.dataframe(cooks_df, use_container_width=True)
            issues.append({
                "level": "warning",
                "msg": f"{n_influential} influential observations detected via Cook's Distance.",
                "fix": "Inspect these rows. Consider removing them if they are data errors, or use Robust Regression.",
            })
    except Exception:
        pass

    # Normality of residuals
    try:
        _, sw_pval = stats.shapiro(result.resid) if len(result.resid) <= 5000 else (None, None)
        if sw_pval is not None:
            st.markdown(f"**Shapiro-Wilk normality test** — p-value: `{sw_pval:.4f}`")
            if sw_pval < 0.05:
                issues.append({
                    "level": "warning",
                    "msg": f"Residuals may not be normally distributed (Shapiro-Wilk p = {sw_pval:.4f})",
                    "fix": "With large samples, minor non-normality is usually not a concern. "
                           "Consider a Box-Cox or log transformation of the outcome.",
                })
    except Exception:
        pass

    st.markdown("### 🩺 Diagnostic Summary")
    show_diagnostics_header(issues)

    # Full summary
    with st.expander("Full Model Summary"):
        st.text(result.summary().as_text())


# ============================================================
# Binary Logistic Regression
# ============================================================

def run_logistic_regression(df, target, predictors, plot_template):
    model_df = prepare_data(df, target, predictors)
    if model_df.empty:
        st.error("No data remaining after dropping missing values.")
        return

    unique_vals = model_df[target].dropna().unique()
    if len(unique_vals) != 2:
        st.error("Binary Logistic Regression requires exactly 2 unique values in the target.")
        return

    # Encode to 0/1 if needed
    if not set(unique_vals).issubset({0, 1}):
        sorted_cls = sorted(unique_vals)
        model_df[target] = model_df[target].map({sorted_cls[0]: 0, sorted_cls[1]: 1})
        st.info(f"Target encoded: {sorted_cls[0]} → 0, {sorted_cls[1]} → 1")

    formula = build_formula(target, predictors, model_df)
    result = smf.logit(formula=formula, data=model_df).fit(disp=False)

    st.success("✅ Model fitted successfully.")
    st.code(formula, language="text")

    # ── Fit metrics ─────────────────────────────────────────
    # McFadden pseudo R²
    mcfadden = 1 - (result.llf / result.llnull)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N", int(result.nobs))
    c2.metric("McFadden R²", round(mcfadden, 4))
    c3.metric("AIC", round(result.aic, 2))
    c4.metric("Log-Likelihood", round(result.llf, 2))

    # ── Coefficients with OR ─────────────────────────────────
    st.markdown("### Coefficients + Odds Ratios")
    tbl = coef_table(result, "Binary Logistic Regression")
    st.dataframe(tbl, use_container_width=True)
    download_buttons(result, tbl, "Binary Logistic Regression")

    # ── Predictions & evaluation ─────────────────────────────
    probs = result.predict(model_df)
    preds_class = (probs >= 0.5).astype(int)
    y_true = model_df[target]

    acc = accuracy_score(y_true, preds_class)
    st.metric("Accuracy", round(acc, 4))

    col_a, col_b = st.columns(2)
    with col_a:
        # Confusion matrix
        cm = confusion_matrix(y_true, preds_class)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
        fig_cm = px.imshow(
            cm_df, text_auto=True, aspect="auto",
            title="Confusion Matrix", color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_b:
        # ROC
        try:
            auc = roc_auc_score(y_true, probs)
            fpr, tpr, _ = roc_curve(y_true, probs)
            fig_roc = px.line(
                x=fpr, y=tpr,
                labels={"x": "False Positive Rate", "y": "True Positive Rate"},
                title=f"ROC Curve (AUC = {auc:.4f})",
                template=plot_template,
            )
            fig_roc.add_shape(
                type="line", x0=0, y0=0, x1=1, y1=1,
                line=dict(dash="dash", color="grey"),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        except Exception:
            st.info("ROC curve could not be generated.")

    # Classification report
    st.markdown("#### Classification Report")
    st.dataframe(
        pd.DataFrame(classification_report(y_true, preds_class, output_dict=True)).T,
        use_container_width=True,
    )

    # Calibration plot
    st.markdown("#### Calibration Plot")
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_mid = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_pred, mean_true = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            mean_pred.append(probs[mask].mean())
            mean_true.append(y_true[mask].mean())
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(
        x=mean_pred, y=mean_true, mode="markers+lines", name="Calibration",
    ))
    fig_cal.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(dash="dash", color="red"),
    )
    fig_cal.update_layout(
        title="Calibration Plot",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        template=plot_template,
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    # ── Diagnostics ──────────────────────────────────────────
    st.markdown("### 🔬 Diagnostics")
    issues = []

    # Class balance
    class_counts = y_true.value_counts()
    minority_pct = class_counts.min() / len(y_true) * 100
    st.markdown(
        f"**Class balance:** {class_counts.to_dict()} "
        f"(minority class = {minority_pct:.1f}%)"
    )
    if minority_pct < 20:
        issues.append({
            "level": "error",
            "msg": f"Class imbalance — minority class is only {minority_pct:.1f}% of data.",
            "fix": "Use class_weight='balanced', oversample with SMOTE, "
                   "or use Penalized Logistic Regression (Firth method).",
        })

    # Hosmer-Lemeshow test
    try:
        g = 10
        quantiles = pd.qcut(probs, g, duplicates="drop")
        hl_df = pd.DataFrame({"obs": y_true.values, "pred": probs.values, "bin": quantiles})
        hl_tbl = hl_df.groupby("bin").agg(
            obs_1=("obs", "sum"),
            pred_1=("pred", "sum"),
            n=("obs", "count"),
        )
        hl_tbl["obs_0"] = hl_tbl["n"] - hl_tbl["obs_1"]
        hl_tbl["pred_0"] = hl_tbl["n"] - hl_tbl["pred_1"]
        hl_stat = (
            ((hl_tbl["obs_1"] - hl_tbl["pred_1"]) ** 2 / hl_tbl["pred_1"]).sum()
            + ((hl_tbl["obs_0"] - hl_tbl["pred_0"]) ** 2 / hl_tbl["pred_0"]).sum()
        )
        hl_pval = 1 - stats.chi2.cdf(hl_stat, df=g - 2)
        st.markdown(
            f"**Hosmer-Lemeshow test** — χ² = `{hl_stat:.4f}`, "
            f"p-value = `{hl_pval:.4f}` (good fit if p > 0.05)"
        )
        if hl_pval < 0.05:
            issues.append({
                "level": "warning",
                "msg": f"Hosmer-Lemeshow test suggests poor calibration (p = {hl_pval:.4f}).",
                "fix": "Add non-linear terms, interaction effects, or additional predictors.",
            })
    except Exception:
        pass

    # McFadden R² guidance
    if mcfadden < 0.1:
        issues.append({
            "level": "warning",
            "msg": f"McFadden R² = {mcfadden:.4f} — model has low explanatory power.",
            "fix": "Consider adding more relevant predictors or interaction terms.",
        })

    # VIF
    try:
        num_preds = model_df[predictors].select_dtypes(include=np.number).columns.tolist()
        if len(num_preds) >= 2:
            X_vif = sm.add_constant(model_df[num_preds].dropna())
            vif_data = pd.DataFrame({
                "Variable": num_preds,
                "VIF": [
                    variance_inflation_factor(X_vif.values, i + 1)
                    for i in range(len(num_preds))
                ],
            }).round(3)
            st.markdown("#### VIF (numeric predictors)")
            st.dataframe(vif_data, use_container_width=True)
            high_vif = vif_data[vif_data["VIF"] > 10]
            if not high_vif.empty:
                issues.append({
                    "level": "error",
                    "msg": f"Multicollinearity (VIF > 10): {', '.join(high_vif['Variable'].tolist())}",
                    "fix": "Remove or combine correlated predictors.",
                })
    except Exception:
        pass

    st.markdown("### 🩺 Diagnostic Summary")
    show_diagnostics_header(issues)

    with st.expander("Full Model Summary"):
        st.text(result.summary().as_text())


# ============================================================
# Poisson Regression
# ============================================================

def run_poisson_regression(df, target, predictors, plot_template):
    model_df = prepare_data(df, target, predictors)
    if model_df.empty:
        st.error("No data remaining after dropping missing values.")
        return

    if not pd.api.types.is_numeric_dtype(model_df[target]):
        st.error("Poisson Regression requires a numeric count outcome.")
        return
    if (model_df[target] < 0).any():
        st.error("Poisson Regression requires non-negative values.")
        return

    mean_t = model_df[target].mean()
    var_t = model_df[target].var()
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean", round(mean_t, 3))
    c2.metric("Variance", round(var_t, 3))
    c3.metric("Var/Mean ratio", round(var_t / mean_t, 3) if mean_t > 0 else "—")

    if var_t > mean_t * 1.5:
        st.warning(
            "⚠️ Variance >> Mean — Overdispersion likely. "
            "Consider Negative Binomial Regression instead."
        )

    formula = build_formula(target, predictors, model_df)
    result = smf.glm(
        formula=formula, data=model_df, family=sm.families.Poisson()
    ).fit()

    st.success("✅ Model fitted successfully.")
    st.code(formula, language="text")

    c1, c2, c3 = st.columns(3)
    c1.metric("N", int(result.nobs))
    c2.metric("AIC", round(result.aic, 2))
    c3.metric("Deviance", round(result.deviance, 2))

    st.markdown("### Coefficients + IRR")
    tbl = coef_table(result, "Poisson Regression")
    st.dataframe(tbl, use_container_width=True)
    download_buttons(result, tbl, "Poisson Regression")

    # Predictions
    preds = result.predict(model_df)
    fig = px.scatter(
        x=model_df[target], y=preds,
        labels={"x": "Actual Count", "y": "Predicted Count"},
        title="Actual vs Predicted Count",
        template=plot_template,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Diagnostics ──────────────────────────────────────────
    st.markdown("### 🔬 Diagnostics")
    issues = []

    # Pearson dispersion
    pearson_chi2 = result.pearson_chi2
    disp_ratio = pearson_chi2 / result.df_resid
    st.markdown(
        f"**Pearson χ² / df = `{disp_ratio:.4f}`** "
        f"(value >> 1 → overdispersion, << 1 → underdispersion)"
    )
    if disp_ratio > 1.5:
        issues.append({
            "level": "error",
            "msg": f"Overdispersion detected (Pearson χ²/df = {disp_ratio:.3f})",
            "fix": "Switch to Negative Binomial Regression. "
                   "If excess zeros exist, consider Zero-Inflated Poisson (ZIP).",
        })

    # Zero inflation check
    zero_pct = (model_df[target] == 0).mean() * 100
    expected_zero_pct = np.exp(-mean_t) * 100
    st.markdown(
        f"**Zero inflation check:** Observed zeros = {zero_pct:.1f}%, "
        f"Expected (Poisson) = {expected_zero_pct:.1f}%"
    )
    if zero_pct > expected_zero_pct * 1.5:
        issues.append({
            "level": "warning",
            "msg": f"Excess zeros detected ({zero_pct:.1f}% vs expected {expected_zero_pct:.1f}%)",
            "fix": "Consider Zero-Inflated Poisson (ZIP) model.",
        })

    st.markdown("### 🩺 Diagnostic Summary")
    show_diagnostics_header(issues)

    with st.expander("Full Model Summary"):
        st.text(result.summary().as_text())


# ============================================================
# Negative Binomial Regression
# ============================================================

def run_negative_binomial(df, target, predictors, plot_template):
    model_df = prepare_data(df, target, predictors)
    if model_df.empty:
        st.error("No data remaining after dropping missing values.")
        return

    if not pd.api.types.is_numeric_dtype(model_df[target]):
        st.error("Negative Binomial Regression requires a numeric count outcome.")
        return
    if (model_df[target] < 0).any():
        st.error("Negative Binomial Regression requires non-negative values.")
        return

    formula = build_formula(target, predictors, model_df)
    result = smf.glm(
        formula=formula, data=model_df, family=sm.families.NegativeBinomial()
    ).fit()

    st.success("✅ Model fitted successfully.")
    st.code(formula, language="text")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N", int(result.nobs))
    c2.metric("AIC", round(result.aic, 2))
    c3.metric("BIC", round(result.bic, 2))
    c4.metric("Deviance", round(result.deviance, 2))

    st.markdown("### Coefficients + IRR")
    tbl = coef_table(result, "Negative Binomial Regression")
    st.dataframe(tbl, use_container_width=True)
    download_buttons(result, tbl, "Negative Binomial Regression")

    preds = result.predict(model_df)
    fig = px.scatter(
        x=model_df[target], y=preds,
        labels={"x": "Actual Count", "y": "Predicted Count"},
        title="Actual vs Predicted Count",
        template=plot_template,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Diagnostics ──────────────────────────────────────────
    st.markdown("### 🔬 Diagnostics")
    issues = []

    disp_ratio = result.pearson_chi2 / result.df_resid
    st.markdown(f"**Pearson χ² / df = `{disp_ratio:.4f}`**")
    if disp_ratio < 0.5:
        issues.append({
            "level": "warning",
            "msg": f"Underdispersion detected (χ²/df = {disp_ratio:.3f}). NB may be over-specified.",
            "fix": "Consider Poisson Regression which may be sufficient.",
        })

    zero_pct = (model_df[target] == 0).mean() * 100
    mean_t = model_df[target].mean()
    expected_zero_pct = np.exp(-mean_t) * 100
    if zero_pct > expected_zero_pct * 1.5:
        issues.append({
            "level": "warning",
            "msg": f"Excess zeros detected ({zero_pct:.1f}%)",
            "fix": "Consider Zero-Inflated Negative Binomial (ZINB) model.",
        })

    st.markdown("### 🩺 Diagnostic Summary")
    show_diagnostics_header(issues)

    with st.expander("Full Model Summary"):
        st.text(result.summary().as_text())


# ============================================================
# Dispatcher — called from Tab 5
# ============================================================

MODEL_OPTIONS = [
    "Linear Regression",
    "Binary Logistic Regression",
    "Poisson Regression",
    "Negative Binomial Regression",
]

MODEL_GUIDANCE = {
    "Linear Regression": (
        "Use when the outcome is **continuous numeric** "
        "(e.g. length of stay, BMI, cost, score)."
    ),
    "Binary Logistic Regression": (
        "Use when the outcome has **exactly two categories** "
        "(e.g. Yes/No, 0/1, readmitted/not readmitted)."
    ),
    "Poisson Regression": (
        "Use when the outcome is a **count** "
        "(e.g. number of visits, admissions, events)."
    ),
    "Negative Binomial Regression": (
        "Use for **count data with overdispersion** "
        "(variance >> mean, common in healthcare counts)."
    ),
}


def render_base_model_tab(df, df_cleaned, plot_template):
    """Called from app.py inside Tab 5."""
    st.markdown("## Statistical Modeling — Base Models")
    st.info(
        "Rows with missing values in the selected variables are removed before fitting. "
        "All diagnostics and fix suggestions appear on this page."
    )

    dataset_choice = st.radio(
        "Dataset to use",
        ["Original data", "Cleaned data (from Data Cleaning tab)"],
        horizontal=True,
    )
    mdf = df_cleaned.copy() if dataset_choice.startswith("Cleaned") else df.copy()

    all_cols = mdf.columns.tolist()
    if len(all_cols) < 2:
        st.warning("Need at least 2 columns to build a model.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        target = st.selectbox("Target / outcome variable", all_cols, key="bm_target")
    with col_b:
        model_type = st.selectbox("Model type", MODEL_OPTIONS, key="bm_type")

    st.info(MODEL_GUIDANCE[model_type])

    available = [c for c in all_cols if c != target]
    predictors = st.multiselect(
        "Predictor variables",
        available,
        default=available[:min(5, len(available))],
        key="bm_predictors",
    )

    if st.button("▶ Run Model", use_container_width=True, key="bm_run"):
        if not predictors:
            st.error("Select at least one predictor.")
            return
        try:
            if model_type == "Linear Regression":
                run_linear_regression(mdf, target, predictors, plot_template)
            elif model_type == "Binary Logistic Regression":
                run_logistic_regression(mdf, target, predictors, plot_template)
            elif model_type == "Poisson Regression":
                run_poisson_regression(mdf, target, predictors, plot_template)
            elif model_type == "Negative Binomial Regression":
                run_negative_binomial(mdf, target, predictors, plot_template)
        except Exception as e:
            st.error(f"Model error: {e}")
            with st.expander("Error details"):
                import traceback
                st.code(traceback.format_exc())

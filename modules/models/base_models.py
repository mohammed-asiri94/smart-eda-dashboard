# ============================================================
# modules/models/base_models.py
# Linear, Logistic, Poisson, Negative Binomial
# Each with full diagnostics + fix suggestions
# Linear Regression diagnostics fixes version: HC3 fix + outcome transforms + centered squared terms
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
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy import stats
from sklearn.model_selection import train_test_split


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


def build_linear_formula(target, predictors, df, polynomial_terms=None):
    """Build an OLS formula with optional squared terms for numeric predictors."""
    polynomial_terms = polynomial_terms or []
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    parts = []
    for c in predictors:
        if c in cat_cols:
            parts.append(f"C({quote_col(c)})")
        else:
            parts.append(quote_col(c))
            if c in polynomial_terms:
                parts.append(f"I({quote_col(c)} ** 2)")
    return f"{quote_col(target)} ~ {' + '.join(parts)}"


def apply_outcome_transformation(model_df, target, outcome_transform):
    """Create transformed outcome when requested and return df, modeling target, and note."""
    transform = outcome_transform or "None"
    if transform == "None":
        return model_df, target, "Outcome is modeled on the original scale."

    df_out = model_df.copy()
    new_target = "__linear_regression_outcome__"

    if transform == "log(y)":
        if (df_out[target] <= 0).any():
            raise ValueError("log(y) requires all outcome values to be greater than 0.")
        df_out[new_target] = np.log(df_out[target])
        return df_out, new_target, "Outcome is modeled as log(y). Coefficients are on the log scale."

    if transform == "sqrt(y)":
        if (df_out[target] < 0).any():
            raise ValueError("sqrt(y) requires all outcome values to be 0 or greater.")
        df_out[new_target] = np.sqrt(df_out[target])
        return df_out, new_target, "Outcome is modeled as sqrt(y). Coefficients are on the square-root scale."

    return model_df, target, "Outcome is modeled on the original scale."




def _safe_internal_name(prefix, col, existing_cols):
    """Create a safe internal column name that will not collide with existing data columns."""
    safe = "".join(ch if str(ch).isalnum() else "_" for ch in str(col)).strip("_")
    if not safe:
        safe = "var"
    base = f"__{prefix}_{safe}__"
    name = base
    i = 1
    while name in existing_cols:
        name = f"{base}_{i}"
        i += 1
    return name


def add_polynomial_features(model_df, polynomial_terms=None, center=True):
    """
    Add optional squared terms for numeric predictors.
    Centering is recommended because x and x^2 are often highly correlated,
    which can inflate VIF and make interpretation unstable.
    """
    polynomial_terms = polynomial_terms or []
    df_out = model_df.copy()
    created_cols = []
    notes = []

    for col in polynomial_terms:
        if col not in df_out.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df_out[col]):
            continue
        new_col = _safe_internal_name("sq", col, set(df_out.columns) | set(created_cols))
        if center:
            center_value = float(df_out[col].mean())
            df_out[new_col] = (df_out[col] - center_value) ** 2
            notes.append(f"Added centered squared term for {col}: ({col} - {center_value:.4f})^2")
        else:
            df_out[new_col] = df_out[col] ** 2
            notes.append(f"Added squared term for {col}: {col}^2")
        created_cols.append(new_col)

    return df_out, created_cols, notes

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

def _as_array(x):
    """Return a flat numpy array from pandas/numpy result attributes."""
    return np.asarray(x).ravel()


def _result_term_names(result):
    """Get coefficient names from statsmodels result objects, including robust results."""
    try:
        return list(result.params.index)
    except Exception:
        try:
            return list(result.model.exog_names)
        except Exception:
            return ["Term " + str(i) for i in range(len(_as_array(result.params)))]


def _linear_coef_table_from_result(result, label="Standard"):
    """Coefficient table for OLS-like results. Works for regular and robust results."""
    params = _as_array(result.params)
    bse = _as_array(result.bse)
    pvals = _as_array(result.pvalues)
    conf = np.asarray(result.conf_int())
    if conf.ndim == 2 and conf.shape[1] >= 2:
        ci_lo = conf[:, 0]
        ci_hi = conf[:, 1]
    else:
        ci_lo = np.full(len(params), np.nan)
        ci_hi = np.full(len(params), np.nan)
    return pd.DataFrame({
        "Term": _result_term_names(result),
        "Coefficient": np.round(params, 4),
        f"{label} Std Error": np.round(bse, 4),
        f"{label} P-value": np.round(pvals, 5),
        "CI Lower": np.round(ci_lo, 4),
        "CI Upper": np.round(ci_hi, 4),
    })


def _interpret_linear_coefficients(tbl):
    """Create simple human-readable interpretations for OLS coefficients."""
    rows = []
    for _, row in tbl.iterrows():
        term = str(row["Term"])
        if term.lower() in ("intercept", "const"):
            continue
        coef = float(row["Coefficient"])
        pval_col = "P-value" if "P-value" in tbl.columns else [c for c in tbl.columns if "P-value" in c][0]
        pval = float(row[pval_col])
        direction = "increase" if coef > 0 else "decrease"
        significance = "statistically significant" if pval < 0.05 else "not statistically significant"
        rows.append({
            "Term": term,
            "Plain interpretation": (
                f"Holding other predictors constant, a one-unit increase/change in {term} "
                f"is associated with a {direction} of {abs(coef):.4f} in the outcome. "
                f"This association is {significance} (p={pval:.4f})."
            ),
        })
    return pd.DataFrame(rows)


def _make_diagnostic_row(check, test, result_text, status, action):
    return {
        "Check": check,
        "Test / Plot": test,
        "Result": result_text,
        "Status": status,
        "Suggested action": action,
    }


def _plot_linear_diagnostics(result, residuals, fitted, plot_template):
    """Four classic OLS diagnostic plots for linear regression."""
    st.markdown("### Regression Diagnostic Plots")
    st.caption(
        "These plots check linearity, normality, equal variance, leverage, and influential observations."
    )

    influence = OLSInfluence(result)
    std_resid = np.asarray(influence.resid_studentized_internal)
    leverage = np.asarray(influence.hat_matrix_diag)
    cooks_d = np.asarray(influence.cooks_distance[0])
    n = len(residuals)
    cooks_threshold = 4 / max(n, 1)

    col_a, col_b = st.columns(2)

    with col_a:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=fitted, y=residuals, mode="markers",
            marker=dict(size=5, opacity=0.60), name="Residuals",
        ))
        try:
            lowess = sm.nonparametric.lowess(residuals, fitted, frac=max(0.25, min(0.8, 30 / max(n, 1))))
            fig1.add_trace(go.Scatter(
                x=lowess[:, 0], y=lowess[:, 1], mode="lines", name="LOWESS trend",
            ))
        except Exception:
            pass
        fig1.add_hline(y=0, line_dash="dash")
        fig1.update_layout(
            title="Residuals vs Fitted",
            xaxis_title="Fitted values",
            yaxis_title="Residuals",
            template=plot_template,
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        osm, osr = stats.probplot(std_resid)[0]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=list(osm), y=list(osr), mode="markers", name="Residuals"))
        fig2.add_trace(go.Scatter(
            x=[min(osm), max(osm)], y=[min(osm), max(osm)],
            mode="lines", line=dict(dash="dash"), name="Normal line",
        ))
        fig2.update_layout(
            title="Normal Q-Q Plot",
            xaxis_title="Theoretical quantiles",
            yaxis_title="Standardized residuals",
            template=plot_template,
        )
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        sqrt_std_resid = np.sqrt(np.abs(std_resid))
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=fitted, y=sqrt_std_resid, mode="markers",
            marker=dict(size=5, opacity=0.60), name="sqrt(|standardized residuals|)",
        ))
        try:
            lowess3 = sm.nonparametric.lowess(sqrt_std_resid, fitted, frac=max(0.25, min(0.8, 30 / max(n, 1))))
            fig3.add_trace(go.Scatter(x=lowess3[:, 0], y=lowess3[:, 1], mode="lines", name="LOWESS trend"))
        except Exception:
            pass
        fig3.update_layout(
            title="Scale-Location",
            xaxis_title="Fitted values",
            yaxis_title="sqrt(|standardized residuals|)",
            template=plot_template,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        point_size = np.clip(cooks_d / max(cooks_threshold, 1e-12) * 8, 5, 25)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=leverage, y=std_resid, mode="markers",
            marker=dict(size=point_size, opacity=0.70),
            text=[f"Cook's D={v:.4f}" for v in cooks_d],
            name="Observations",
        ))
        fig4.add_hline(y=0, line_dash="dash")
        fig4.update_layout(
            title="Residuals vs Leverage",
            xaxis_title="Leverage / hat value",
            yaxis_title="Standardized residuals",
            template=plot_template,
        )
        st.plotly_chart(fig4, use_container_width=True)


def _inverse_outcome_transform(values, outcome_transform):
    """Back-transform predictions to the original outcome scale when possible."""
    arr = np.asarray(values, dtype=float)
    if outcome_transform == "log(y)":
        return np.exp(arr)
    if outcome_transform == "sqrt(y)":
        return np.square(arr)
    return arr


def _add_polynomial_columns_to_new_data(new_df, poly_specs):
    """Create the same squared-term columns in new data using training means."""
    df_new = new_df.copy()
    for spec in poly_specs:
        source = spec["source"]
        feature = spec["feature"]
        if source not in df_new.columns:
            raise ValueError(f"New data is missing required predictor column: {source}")
        vals = pd.to_numeric(df_new[source], errors="coerce")
        if spec.get("center", True):
            df_new[feature] = (vals - spec.get("mean", 0.0)) ** 2
        else:
            df_new[feature] = vals ** 2
    return df_new


def _prediction_split_section(
    model_df,
    original_target,
    model_target,
    original_predictors,
    formula,
    full_result,
    test_size,
    random_state,
    plot_template,
    outcome_transform="None",
    poly_specs=None,
):
    """Train/test validation and new-data prediction for prediction-oriented linear regression."""
    poly_specs = poly_specs or []
    st.markdown("## Prediction / Validation Workflow")
    st.info(
        "Prediction mode trains the model on a training set and evaluates it on a held-out test set. "
        "The main question is: how well does the model predict unseen data?"
    )

    if len(model_df) < 20:
        st.warning("Train/test split is not reliable with very small datasets. Use at least 20 observations, preferably more.")
        return

    try:
        train_df, test_df = train_test_split(
            model_df,
            test_size=float(test_size),
            random_state=int(random_state),
        )
        train_result = smf.ols(formula=formula, data=train_df).fit()

        train_pred_model_scale = train_result.predict(train_df)
        test_pred_model_scale = train_result.predict(test_df)

        y_train_actual = train_df[original_target].astype(float).values
        y_test_actual = test_df[original_target].astype(float).values
        train_pred = _inverse_outcome_transform(train_pred_model_scale, outcome_transform)
        test_pred = _inverse_outcome_transform(test_pred_model_scale, outcome_transform)

        train_r2 = float(r2_score(y_train_actual, train_pred))
        test_r2 = float(r2_score(y_test_actual, test_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test_actual, test_pred)))
        mae = float(mean_absolute_error(y_test_actual, test_pred))

        st.markdown("### Test-set Performance")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Train N", len(train_df))
        c2.metric("Test N", len(test_df))
        c3.metric("Train R2", round(train_r2, 4))
        c4.metric("Test R2", round(test_r2, 4))
        c5.metric("Test RMSE", round(rmse, 4))
        st.metric("Test MAE", round(mae, 4))

        if test_r2 < 0:
            st.warning(
                "Test R2 is negative. This means the model predicts worse than simply using the test-set mean. "
                "Consider adding relevant predictors, non-linear terms, or checking data quality."
            )
        elif train_r2 - test_r2 > 0.15:
            st.warning(
                "Train R2 is much higher than Test R2. This may indicate overfitting or unstable prediction."
            )
        else:
            st.success("Train and test performance are reasonably consistent.")

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.scatter(
                x=y_test_actual,
                y=test_pred,
                labels={"x": "Actual test values", "y": "Predicted test values"},
                title="Test Set: Actual vs Predicted",
                template=plot_template,
            )
            min_v = min(float(np.nanmin(y_test_actual)), float(np.nanmin(test_pred)))
            max_v = max(float(np.nanmax(y_test_actual)), float(np.nanmax(test_pred)))
            fig.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(dash="dash"))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            test_resid = y_test_actual - test_pred
            fig_res = px.scatter(
                x=test_pred,
                y=test_resid,
                labels={"x": "Predicted test values", "y": "Prediction error"},
                title="Test Set: Prediction Error vs Predicted",
                template=plot_template,
            )
            fig_res.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig_res, use_container_width=True)

        pred_tbl = test_df[original_predictors].copy()
        pred_tbl["Actual"] = np.round(y_test_actual, 4)
        pred_tbl["Predicted"] = np.round(test_pred, 4)
        pred_tbl["Error"] = np.round(y_test_actual - test_pred, 4)
        pred_tbl["Absolute Error"] = np.round(np.abs(y_test_actual - test_pred), 4)

        with st.expander("Optional: Test-set prediction table and download", expanded=False):
            st.caption("Shows the first 50 rows from the held-out test set. Use the download button for the full table.")
            st.dataframe(pred_tbl.head(50), use_container_width=True)
            st.download_button(
                "Download test-set predictions (CSV)",
                data=pred_tbl.to_csv(index=False).encode(),
                file_name="linear_regression_test_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("### Predict on New Data")
        st.caption(
            "After validation, the app uses the full-data model to predict new observations. "
            "New data must contain the same predictor columns used in the model."
        )

        with st.expander("Manual single prediction", expanded=False):
            manual_values = {}
            cols = st.columns(2)
            for i, pred in enumerate(original_predictors):
                with cols[i % 2]:
                    if pd.api.types.is_numeric_dtype(model_df[pred]):
                        default_val = float(pd.to_numeric(model_df[pred], errors="coerce").median())
                        manual_values[pred] = st.number_input(
                            pred,
                            value=default_val,
                            key=f"lr_manual_pred_{pred}",
                        )
                    else:
                        vals = model_df[pred].dropna().astype(str).unique().tolist()
                        vals = sorted(vals) if vals else [""]
                        manual_values[pred] = st.selectbox(
                            pred,
                            vals,
                            key=f"lr_manual_pred_{pred}",
                        )
            if st.button("Predict single row", key="lr_predict_single", use_container_width=True):
                try:
                    one_df = pd.DataFrame([manual_values])
                    # Cast categorical-like columns back to object strings to match formula categories.
                    for pred in original_predictors:
                        if not pd.api.types.is_numeric_dtype(model_df[pred]):
                            one_df[pred] = one_df[pred].astype(str)
                    one_df = _add_polynomial_columns_to_new_data(one_df, poly_specs)
                    pred_model_scale = full_result.predict(one_df)
                    pred_original = _inverse_outcome_transform(pred_model_scale, outcome_transform)[0]
                    st.success(f"Predicted {original_target}: {pred_original:.4f}")
                except Exception as e:
                    st.error("Prediction failed: " + str(e))

        uploaded_new = st.file_uploader(
            "Upload new data for prediction (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            key="lr_new_prediction_file",
        )
        if uploaded_new is not None:
            try:
                if uploaded_new.name.lower().endswith(".csv"):
                    new_df = pd.read_csv(uploaded_new)
                else:
                    new_df = pd.read_excel(uploaded_new)

                missing_cols = [c for c in original_predictors if c not in new_df.columns]
                if missing_cols:
                    st.error("New data is missing required columns: " + ", ".join(missing_cols))
                else:
                    pred_input = new_df.copy()
                    pred_input = _add_polynomial_columns_to_new_data(pred_input, poly_specs)
                    new_pred_model_scale = full_result.predict(pred_input)
                    new_pred = _inverse_outcome_transform(new_pred_model_scale, outcome_transform)
                    output_df = new_df.copy()
                    output_df[f"predicted_{original_target}"] = np.round(new_pred, 4)
                    st.dataframe(output_df.head(50), use_container_width=True)
                    st.download_button(
                        "Download new-data predictions (CSV)",
                        data=output_df.to_csv(index=False).encode(),
                        file_name=f"predicted_{original_target}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error("New-data prediction failed: " + str(e))

    except Exception as e:
        st.warning("Train/test prediction workflow could not be completed: " + str(e))


def _quick_prediction_health_check(result, model_df, train_r2=None, test_r2=None):
    """Compact health checks for prediction mode only.

    These checks keep prediction output short. Full OLS diagnostics remain available
    through an optional checkbox.
    """
    st.markdown("### Quick Prediction Health Check")
    quick_rows = []

    # Multicollinearity: high VIF can make predictions unstable.
    try:
        exog = pd.DataFrame(result.model.exog, columns=result.model.exog_names)
        exog = exog.loc[:, exog.nunique(dropna=True) > 1]
        exog_no_intercept = exog.drop(columns=["Intercept"], errors="ignore")

        if exog_no_intercept.shape[1] >= 2:
            X_vif = sm.add_constant(exog_no_intercept, has_constant="add")
            vif_values = []
            for i, col in enumerate(X_vif.columns):
                if col == "const":
                    continue
                try:
                    vif_values.append(variance_inflation_factor(X_vif.values, i))
                except Exception:
                    vif_values.append(np.nan)
            finite_vif = [v for v in vif_values if np.isfinite(v)]
            max_vif = float(np.max(finite_vif)) if finite_vif else np.nan

            if np.isfinite(max_vif) and max_vif > 10:
                status = "Problem"
                action = "High VIF. Predictions may be unstable; remove/combine correlated predictors."
            elif np.isfinite(max_vif) and max_vif > 5:
                status = "Warning"
                action = "Moderate VIF. Review correlated predictors."
            else:
                status = "OK"
                action = "No major multicollinearity issue."

            quick_rows.append({
                "Check": "Multicollinearity",
                "Result": f"Max VIF = {max_vif:.3f}" if np.isfinite(max_vif) else "Not available",
                "Status": status,
                "Action": action,
            })
        else:
            quick_rows.append({
                "Check": "Multicollinearity",
                "Result": "Not enough encoded predictors for VIF",
                "Status": "Not assessed",
                "Action": "Usually acceptable for simple prediction models.",
            })
    except Exception as e:
        quick_rows.append({
            "Check": "Multicollinearity",
            "Result": "VIF failed",
            "Status": "Not assessed",
            "Action": f"Reason: {e}",
        })

    # Heteroscedasticity: more important for inference, but useful warning.
    try:
        _, bp_pval, _, _ = het_breuschpagan(result.resid, result.model.exog)
        if bp_pval < 0.05:
            status = "Warning"
            action = "Unequal variance detected. Prediction may still be usable; use HC3 mainly if interpreting coefficients."
        else:
            status = "OK"
            action = "No strong unequal-variance signal."
        quick_rows.append({
            "Check": "Error variance",
            "Result": f"Breusch-Pagan p = {bp_pval:.4f}",
            "Status": status,
            "Action": action,
        })
    except Exception as e:
        quick_rows.append({
            "Check": "Error variance",
            "Result": "Breusch-Pagan failed",
            "Status": "Not assessed",
            "Action": f"Reason: {e}",
        })

    # Influential observations: a few points can dominate a linear prediction model.
    try:
        influence = OLSInfluence(result)
        cooks_d = np.asarray(influence.cooks_distance[0])
        threshold = 4 / max(len(model_df), 1)
        n_influential = int((cooks_d > threshold).sum())
        if n_influential > 0:
            status = "Warning"
            action = "Inspect if these are data errors or valid unusual cases."
        else:
            status = "OK"
            action = "No major influential rows detected."
        quick_rows.append({
            "Check": "Influential observations",
            "Result": f"{n_influential} rows above Cook's D threshold",
            "Status": status,
            "Action": action,
        })
    except Exception as e:
        quick_rows.append({
            "Check": "Influential observations",
            "Result": "Cook's Distance failed",
            "Status": "Not assessed",
            "Action": f"Reason: {e}",
        })

    quick_df = pd.DataFrame(quick_rows)
    st.dataframe(quick_df, use_container_width=True)
    st.caption(
        "Prediction mode focuses on Test R², RMSE, MAE, prediction plots, and new-data prediction. "
        "Full OLS diagnostics are optional below."
    )

def run_linear_regression(
    df,
    target,
    predictors,
    plot_template,
    enable_prediction_split=False,
    test_size=0.20,
    split_random_state=42,
    show_robust_hc3=True,
    compare_without_influential=True,
    outcome_transform="None",
    polynomial_terms=None,
    center_polynomial_terms=True,
):
    polynomial_terms = polynomial_terms or []
    model_df = prepare_data(df, target, predictors)
    if model_df.empty:
        st.error("No data remaining after dropping missing values.")
        return

    if len(model_df) < 5:
        st.error("Linear regression needs at least 5 complete observations after removing missing values.")
        return

    try:
        model_df, model_target, transform_note = apply_outcome_transformation(
            model_df, target, outcome_transform
        )
    except Exception as e:
        st.error("Outcome transformation could not be applied: " + str(e))
        return

    model_df, polynomial_feature_cols, polynomial_notes = add_polynomial_features(
        model_df, polynomial_terms=polynomial_terms, center=center_polynomial_terms
    )
    poly_specs = []
    for source_col, feature_col in zip(polynomial_terms, polynomial_feature_cols):
        if source_col in model_df.columns:
            poly_specs.append({
                "source": source_col,
                "feature": feature_col,
                "center": bool(center_polynomial_terms),
                "mean": float(pd.to_numeric(model_df[source_col], errors="coerce").mean()) if center_polynomial_terms else 0.0,
            })
    model_predictors = predictors + polynomial_feature_cols
    formula = build_formula(model_target, model_predictors, model_df)
    result = smf.ols(formula=formula, data=model_df).fit()

    st.success("Model fitted successfully.")
    st.code(formula, language="text")
    if outcome_transform != "None" or polynomial_notes:
        msg = transform_note
        if polynomial_notes:
            msg += " " + " | ".join(polynomial_notes)
            msg += " These terms are used to handle possible non-linearity."
        st.info(msg)

    # Predictions from the full-data model are prepared once.
    preds = result.predict(model_df)
    residuals = model_df[model_target] - preds
    tbl = coef_table(result, "Linear Regression")

    if enable_prediction_split:
        st.markdown("## Prediction Focus Mode")
        st.info(
            "Prediction mode focuses on held-out test-set performance and prediction on new data. "
            "Detailed OLS diagnostics are hidden by default to keep this section concise."
        )

        _prediction_split_section(
            model_df=model_df,
            original_target=target,
            model_target=model_target,
            original_predictors=predictors,
            formula=formula,
            full_result=result,
            test_size=test_size,
            random_state=split_random_state,
            plot_template=plot_template,
            outcome_transform=outcome_transform,
            poly_specs=poly_specs,
        )

        _quick_prediction_health_check(result, model_df)

        show_full_prediction_diagnostics = st.checkbox(
            "Show full regression diagnostics in prediction mode",
            value=False,
            key="lr_show_full_prediction_diagnostics",
            help=(
                "Turn this on only if you want the full OLS diagnostic output. "
                "For prediction, focus mainly on Test R², RMSE, MAE, and prediction errors."
            ),
        )
        if not show_full_prediction_diagnostics:
            st.info(
                "Full regression diagnostics are hidden in prediction mode. "
                "Enable the checkbox above if you want to inspect all OLS assumptions."
            )
            return

        with st.expander("Full-data OLS coefficients and fit plot", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("N", int(result.nobs))
            c2.metric("R2", round(result.rsquared, 4))
            c3.metric("Adjusted R2", round(result.rsquared_adj, 4))
            c4.metric("AIC", round(result.aic, 2))

            st.markdown("### Coefficients")
            st.dataframe(tbl, use_container_width=True)
            download_buttons(result, tbl, "Linear Regression")

            interp_tbl = _interpret_linear_coefficients(tbl)
            if not interp_tbl.empty:
                st.markdown("#### Plain-language coefficient interpretation")
                st.dataframe(interp_tbl, use_container_width=True)

            fig_fit = px.scatter(
                x=model_df[model_target], y=preds,
                labels={"x": "Actual", "y": "Predicted"},
                title="Full Data: Actual vs Predicted",
                template=plot_template,
            )
            min_v = min(float(model_df[model_target].min()), float(preds.min()))
            max_v = max(float(model_df[model_target].max()), float(preds.max()))
            fig_fit.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(dash="dash"))
            st.plotly_chart(fig_fit, use_container_width=True)

    else:
        # Fit metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("N", int(result.nobs))
        c2.metric("R2", round(result.rsquared, 4))
        c3.metric("Adjusted R2", round(result.rsquared_adj, 4))
        c4.metric("AIC", round(result.aic, 2))

        # Coefficients
        st.markdown("### Coefficients")
        st.dataframe(tbl, use_container_width=True)
        download_buttons(result, tbl, "Linear Regression")

        interp_tbl = _interpret_linear_coefficients(tbl)
        if not interp_tbl.empty:
            with st.expander("Plain-language coefficient interpretation"):
                st.dataframe(interp_tbl, use_container_width=True)

        # Actual vs predicted
        st.markdown("### Model Fit Plot")
        fig_fit = px.scatter(
            x=model_df[model_target], y=preds,
            labels={"x": "Actual", "y": "Predicted"},
            title="Actual vs Predicted",
            template=plot_template,
        )
        min_v = min(float(model_df[model_target].min()), float(preds.min()))
        max_v = max(float(model_df[model_target].max()), float(preds.max()))
        fig_fit.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(dash="dash"))
        st.plotly_chart(fig_fit, use_container_width=True)

    # Diagnostics
    if enable_prediction_split:
        st.markdown("### Full Linear Regression Diagnostics")
        st.caption("Optional full diagnostics for users who want to inspect OLS assumptions after prediction evaluation.")
    else:
        st.markdown("### Linear Regression Diagnostics")
    issues = []
    checklist = []

    influence = OLSInfluence(result)
    cooks_d = np.asarray(influence.cooks_distance[0])
    leverage = np.asarray(influence.hat_matrix_diag)
    std_resid_int = np.asarray(influence.resid_studentized_internal)
    try:
        std_resid_ext = np.asarray(influence.resid_studentized_external)
    except Exception:
        std_resid_ext = std_resid_int
    n = len(model_df)
    p_model = int(result.df_model) + 1

    _plot_linear_diagnostics(result, np.asarray(residuals), np.asarray(preds), plot_template)

    # VIF using design matrix after formula encoding, including categorical dummy variables
    try:
        exog = pd.DataFrame(result.model.exog, columns=result.model.exog_names)
        exog = exog.loc[:, exog.nunique(dropna=True) > 1]
        if "Intercept" in exog.columns:
            exog_no_intercept = exog.drop(columns=["Intercept"])
        else:
            exog_no_intercept = exog.copy()

        if exog_no_intercept.shape[1] >= 2:
            X_vif = sm.add_constant(exog_no_intercept, has_constant="add")
            vif_rows = []
            for i, col in enumerate(X_vif.columns):
                if col == "const":
                    continue
                try:
                    vif_val = variance_inflation_factor(X_vif.values, i)
                except Exception:
                    vif_val = np.nan
                vif_rows.append({"Variable": col, "VIF": round(float(vif_val), 3) if np.isfinite(vif_val) else np.nan})
            vif_data = pd.DataFrame(vif_rows)
            st.markdown("#### Multicollinearity: VIF")
            st.caption("VIF is calculated after formula encoding, so categorical predictors are checked through their dummy variables.")
            st.dataframe(vif_data, use_container_width=True)

            max_vif = float(vif_data["VIF"].dropna().max()) if not vif_data["VIF"].dropna().empty else np.nan
            if np.isfinite(max_vif) and max_vif > 10:
                issues.append({
                    "level": "error",
                    "msg": f"Strong multicollinearity detected (max VIF = {max_vif:.2f}).",
                    "fix": "Remove or combine highly correlated predictors, reduce dummy categories, or consider Ridge/Lasso for prediction.",
                })
                vif_status = "Problem"
            elif np.isfinite(max_vif) and max_vif > 5:
                issues.append({
                    "level": "warning",
                    "msg": f"Moderate multicollinearity detected (max VIF = {max_vif:.2f}).",
                    "fix": "Review correlated predictors and consider removing one of them if interpretation is unstable.",
                })
                vif_status = "Warning"
            else:
                vif_status = "Passed"
            checklist.append(_make_diagnostic_row(
                "Multicollinearity", "VIF", f"Max VIF = {max_vif:.3f}" if np.isfinite(max_vif) else "Not available",
                vif_status, "VIF < 5 is usually acceptable; VIF > 10 is a strong problem."
            ))
        else:
            checklist.append(_make_diagnostic_row(
                "Multicollinearity", "VIF", "Not enough predictors for VIF", "Not assessed",
                "VIF requires at least two encoded predictors."
            ))
    except Exception as e:
        checklist.append(_make_diagnostic_row("Multicollinearity", "VIF", "Failed: " + str(e), "Not assessed", "Check selected predictors."))

    # Breusch-Pagan
    bp_pval = None
    try:
        bp_lm, bp_pval, _, _ = het_breuschpagan(result.resid, result.model.exog)
        st.markdown(f"**Breusch-Pagan test** - LM Statistic: `{bp_lm:.4f}`, p-value: `{bp_pval:.4f}`")
        if bp_pval < 0.05:
            issues.append({
                "level": "warning",
                "msg": f"Heteroscedasticity detected (Breusch-Pagan p = {bp_pval:.4f}).",
                "fix": "Use HC3 robust standard errors, transform the outcome, or consider weighted least squares.",
            })
            bp_status = "Problem"
        else:
            bp_status = "Passed"
        checklist.append(_make_diagnostic_row(
            "Homoscedasticity", "Breusch-Pagan", f"p = {bp_pval:.4f}", bp_status,
            "If p < 0.05, use HC3 robust SE or transform the outcome."
        ))
    except Exception as e:
        checklist.append(_make_diagnostic_row("Homoscedasticity", "Breusch-Pagan", "Failed: " + str(e), "Not assessed", "Check model design matrix."))

    # Robust HC3 option
    if show_robust_hc3:
        st.markdown("#### Robust Standard Errors (HC3)")
        st.caption("HC3 keeps the same coefficients but adjusts standard errors and p-values when variance is not constant.")
        try:
            robust_result = result.get_robustcov_results(cov_type="HC3")
            robust_tbl = _linear_coef_table_from_result(robust_result, label="HC3 Robust")
            compare_tbl = tbl[["Term", "Coefficient", "Std Error", "P-value"]].merge(
                robust_tbl[["Term", "HC3 Robust Std Error", "HC3 Robust P-value"]],
                on="Term",
                how="left",
            )
            compare_tbl["Decision changed?"] = (
                (compare_tbl["P-value"] < 0.05) != (compare_tbl["HC3 Robust P-value"] < 0.05)
            ).map({True: "YES", False: "NO"})
            st.dataframe(compare_tbl, use_container_width=True)
            st.download_button(
                "Download robust SE comparison (CSV)",
                data=compare_tbl.to_csv(index=False).encode(),
                file_name="linear_regression_robust_hc3.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.info("Robust HC3 table could not be generated: " + str(e))

    # Durbin-Watson
    try:
        dw = float(durbin_watson(result.resid))
        st.markdown(f"**Durbin-Watson statistic:** `{dw:.4f}` (ideal is approximately 2.0)")
        if dw < 1.5 or dw > 2.5:
            issues.append({
                "level": "warning",
                "msg": f"Possible autocorrelation in residuals (Durbin-Watson = {dw:.4f}).",
                "fix": "If rows are time-ordered, add time/lag terms, use GLS, or use a time-series model.",
            })
            dw_status = "Warning"
        else:
            dw_status = "Passed"
        checklist.append(_make_diagnostic_row(
            "Independence", "Durbin-Watson", f"DW = {dw:.4f}", dw_status,
            "Values close to 2 suggest independent errors."
        ))
    except Exception as e:
        checklist.append(_make_diagnostic_row("Independence", "Durbin-Watson", "Failed: " + str(e), "Not assessed", "Use only when row order is meaningful."))

    # Normality
    try:
        if n <= 5000 and n >= 3:
            _, sw_pval = stats.shapiro(result.resid)
            st.markdown(f"**Shapiro-Wilk normality test** - p-value: `{sw_pval:.4f}`")
            if sw_pval < 0.05:
                issues.append({
                    "level": "warning",
                    "msg": f"Residuals may not be normally distributed (Shapiro-Wilk p = {sw_pval:.4f}).",
                    "fix": "Inspect the Q-Q plot. With large samples this may be minor; otherwise consider outcome transformation or robust methods.",
                })
                norm_status = "Warning"
            else:
                norm_status = "Passed"
            checklist.append(_make_diagnostic_row(
                "Normality of errors", "Q-Q plot + Shapiro-Wilk", f"p = {sw_pval:.4f}", norm_status,
                "If clearly non-normal, inspect outliers and consider log/sqrt/Box-Cox transformation."
            ))
        else:
            checklist.append(_make_diagnostic_row(
                "Normality of errors", "Q-Q plot", "Shapiro skipped for n > 5000", "Visual check",
                "Use Q-Q plot; large samples often flag trivial departures."
            ))
    except Exception as e:
        checklist.append(_make_diagnostic_row("Normality of errors", "Shapiro-Wilk", "Failed: " + str(e), "Not assessed", "Check residuals."))

    # Linearity by Harvey-Collier plus plot guidance
    try:
        hc_stat, hc_p = linear_harvey_collier(result)
        if hc_p < 0.05:
            issues.append({
                "level": "warning",
                "msg": f"Possible non-linearity detected (Harvey-Collier p = {hc_p:.4f}).",
                "fix": "Add polynomial terms, splines, transformations, or interaction terms if scientifically justified.",
            })
            lin_status = "Warning"
        else:
            lin_status = "Passed"
        checklist.append(_make_diagnostic_row(
            "Linearity", "Residual plot + Harvey-Collier", f"p = {hc_p:.4f}", lin_status,
            "If LOWESS curve bends, add polynomial/spline/transform terms."
        ))
    except Exception:
        checklist.append(_make_diagnostic_row(
            "Linearity", "Residuals vs fitted", "Visual check only", "Visual check",
            "Look for random scatter around zero; curved pattern suggests non-linearity."
        ))

    # Cook's distance
    cooks_threshold = 4 / max(n, 1)
    influential_mask = cooks_d > cooks_threshold
    n_influential = int(influential_mask.sum())
    st.markdown(f"**Cook's Distance** - {n_influential} influential observation(s), threshold = `{cooks_threshold:.4f}`")
    if n_influential > 0:
        cooks_df = pd.DataFrame({
            "Original row index": model_df.index[influential_mask],
            "Observation position": np.where(influential_mask)[0],
            "Cook's D": np.round(cooks_d[influential_mask], 5),
            "Leverage": np.round(leverage[influential_mask], 5),
            "Studentized residual": np.round(std_resid_ext[influential_mask], 4),
        }).sort_values("Cook's D", ascending=False)
        st.dataframe(cooks_df, use_container_width=True)
        issues.append({
            "level": "warning",
            "msg": f"{n_influential} influential observation(s) detected by Cook's Distance.",
            "fix": "Inspect these rows. If they are data errors, correct/remove them. If real, report a sensitivity analysis.",
        })
        cook_status = "Warning"
    else:
        cook_status = "Passed"
    checklist.append(_make_diagnostic_row(
        "Influential points", "Cook's Distance", f"{n_influential} above 4/n", cook_status,
        "Inspect influential rows; compare model with and without them."
    ))

    # High leverage
    lev_threshold = 2 * p_model / max(n, 1)
    high_lev_mask = leverage > lev_threshold
    n_high_lev = int(high_lev_mask.sum())
    st.markdown(f"**High leverage / hat values** - {n_high_lev} high-leverage observation(s), threshold = `{lev_threshold:.4f}`")
    if n_high_lev > 0:
        lev_df = pd.DataFrame({
            "Original row index": model_df.index[high_lev_mask],
            "Observation position": np.where(high_lev_mask)[0],
            "Hat value": np.round(leverage[high_lev_mask], 5),
            "Cook's D": np.round(cooks_d[high_lev_mask], 5),
            "Studentized residual": np.round(std_resid_ext[high_lev_mask], 4),
        }).sort_values("Hat value", ascending=False)
        with st.expander("High leverage observations"):
            st.dataframe(lev_df, use_container_width=True)
        issues.append({
            "level": "info",
            "msg": f"{n_high_lev} high-leverage observation(s) detected.",
            "fix": "High leverage means unusual X values. They are not automatically wrong; inspect them and their influence.",
        })
        lev_status = "Warning"
    else:
        lev_status = "Passed"
    checklist.append(_make_diagnostic_row(
        "High leverage", "Hat values", f"{n_high_lev} above 2p/n", lev_status,
        "Inspect unusual predictor combinations; do not remove automatically."
    ))

    # Outlier test using externally studentized residuals + Bonferroni
    try:
        df_resid = max(float(result.df_resid), 1.0)
        outlier_p = 2 * stats.t.sf(np.abs(std_resid_ext), df=df_resid)
        bonf_p = np.minimum(outlier_p * n, 1.0)
        outlier_mask = bonf_p < 0.05
        n_outliers = int(outlier_mask.sum())
        st.markdown(f"**Outlier test** - {n_outliers} observation(s) with Bonferroni p < 0.05")
        if n_outliers > 0:
            outlier_df = pd.DataFrame({
                "Original row index": model_df.index[outlier_mask],
                "Observation position": np.where(outlier_mask)[0],
                "Studentized residual": np.round(std_resid_ext[outlier_mask], 4),
                "Bonferroni p-value": np.round(bonf_p[outlier_mask], 6),
                "Cook's D": np.round(cooks_d[outlier_mask], 5),
            }).sort_values("Bonferroni p-value")
            st.dataframe(outlier_df, use_container_width=True)
            issues.append({
                "level": "warning",
                "msg": f"{n_outliers} possible outcome outlier(s) detected using studentized residuals.",
                "fix": "Check whether these are data entry errors, rare valid outcomes, or signs of model misspecification.",
            })
            out_status = "Warning"
        else:
            out_status = "Passed"
        checklist.append(_make_diagnostic_row(
            "Outliers", "Studentized residuals + Bonferroni", f"{n_outliers} flagged", out_status,
            "Inspect flagged Y-outliers; compare with Cook's D before removing."
        ))
    except Exception as e:
        checklist.append(_make_diagnostic_row("Outliers", "Studentized residuals", "Failed: " + str(e), "Not assessed", "Check residuals."))

    # DFBETAS
    try:
        dfbetas = np.asarray(influence.dfbetas)
        dfbeta_threshold = 2 / np.sqrt(max(n, 1))
        dfbeta_mask = np.abs(dfbetas) > dfbeta_threshold
        n_dfbeta_cells = int(dfbeta_mask.sum())
        n_dfbeta_rows = int(dfbeta_mask.any(axis=1).sum())
        st.markdown(f"**DFBETAS** - {n_dfbeta_rows} observation(s) affect at least one coefficient, threshold = `{dfbeta_threshold:.4f}`")
        if n_dfbeta_rows > 0:
            rows = []
            terms = list(result.params.index)
            for i in np.where(dfbeta_mask.any(axis=1))[0]:
                affected_terms = [terms[j] for j in np.where(dfbeta_mask[i])[0]]
                max_abs = float(np.max(np.abs(dfbetas[i])))
                rows.append({
                    "Original row index": model_df.index[i],
                    "Observation position": int(i),
                    "Max |DFBETAS|": round(max_abs, 5),
                    "Affected coefficients": ", ".join(affected_terms[:6]) + (" ..." if len(affected_terms) > 6 else ""),
                })
            dfbeta_df = pd.DataFrame(rows).sort_values("Max |DFBETAS|", ascending=False)
            with st.expander("DFBETAS influential observations"):
                st.dataframe(dfbeta_df, use_container_width=True)
            issues.append({
                "level": "warning",
                "msg": f"{n_dfbeta_rows} observation(s) have large DFBETAS.",
                "fix": "These observations change one or more coefficients. Inspect them and run sensitivity analysis.",
            })
            dfb_status = "Warning"
        else:
            dfb_status = "Passed"
        checklist.append(_make_diagnostic_row(
            "Coefficient influence", "DFBETAS", f"{n_dfbeta_rows} rows flagged", dfb_status,
            "If flagged, check which coefficient changes and compare a refitted model."
        ))
    except Exception as e:
        checklist.append(_make_diagnostic_row("Coefficient influence", "DFBETAS", "Failed: " + str(e), "Not assessed", "Check model rank and observations."))

    # Refit without influential observations
    if compare_without_influential and n_influential > 0 and n - n_influential > p_model + 2:
        st.markdown("### Sensitivity Analysis: Refit Without Influential Observations")
        try:
            refit_df = model_df.loc[~influential_mask].copy()
            refit_result = smf.ols(formula=formula, data=refit_df).fit()
            compare_metrics = pd.DataFrame({
                "Metric": ["N", "R2", "Adjusted R2", "AIC", "BIC"],
                "Original model": [
                    int(result.nobs),
                    round(result.rsquared, 4),
                    round(result.rsquared_adj, 4),
                    round(result.aic, 2),
                    round(result.bic, 2),
                ],
                "Without influential": [
                    int(refit_result.nobs),
                    round(refit_result.rsquared, 4),
                    round(refit_result.rsquared_adj, 4),
                    round(refit_result.aic, 2),
                    round(refit_result.bic, 2),
                ],
            })
            st.dataframe(compare_metrics, use_container_width=True)

            refit_tbl = coef_table(refit_result, "Linear Regression")
            coef_compare = tbl[["Term", "Coefficient", "P-value"]].merge(
                refit_tbl[["Term", "Coefficient", "P-value"]],
                on="Term",
                suffixes=(" Original", " Without influential"),
                how="outer",
            )
            coef_compare["Coefficient change"] = (
                coef_compare["Coefficient Without influential"] - coef_compare["Coefficient Original"]
            ).round(4)
            st.dataframe(coef_compare, use_container_width=True)
            st.info(
                "Do not remove real observations only because they are influential. "
                "Use this as sensitivity analysis: if conclusions change, report both results and inspect the rows."
            )
        except Exception as e:
            st.warning("Refit comparison failed: " + str(e))

    # Final checklist
    st.markdown("### Final Diagnostic Checklist")
    checklist_df = pd.DataFrame(checklist)
    st.dataframe(checklist_df, use_container_width=True)
    st.download_button(
        "Download diagnostic checklist (CSV)",
        data=checklist_df.to_csv(index=False).encode(),
        file_name="linear_regression_diagnostic_checklist.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("### Problem-handling guide")
    st.info(
        "If a diagnostic fails, choose the fix based on the problem: "
        "non-linearity -> add polynomial/spline/transform terms; "
        "heteroscedasticity -> HC3 robust SE or transform outcome; "
        "non-normal residuals -> inspect outliers/transform outcome; "
        "autocorrelation -> add time/lag terms or use a time-series model; "
        "multicollinearity -> remove/combine correlated predictors; "
        "influential points -> inspect rows and run sensitivity analysis."
    )

    st.markdown("### Diagnostic Summary")
    show_diagnostics_header(issues)

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
        x=model_df[model_target], y=preds,
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
        x=model_df[model_target], y=preds,
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

    # Linear regression has two possible goals:
    # 1) inference/explanation: use full data + assumptions diagnostics
    # 2) prediction: optionally add train/test validation metrics
    lr_enable_split = False
    lr_test_size = 0.20
    lr_random_state = 42
    lr_show_robust = True
    lr_compare_influential = True
    lr_outcome_transform = "None"
    lr_polynomial_terms = []
    lr_center_polynomial_terms = True

    if model_type == "Linear Regression":
        with st.expander("Linear Regression options", expanded=True):
            purpose = st.radio(
                "Analysis purpose",
                [
                    "Statistical inference / explanation",
                    "Prediction / validation",
                ],
                horizontal=True,
                key="lr_purpose",
                help=(
                    "Inference focuses on coefficients, p-values, confidence intervals, and diagnostics. "
                    "Prediction adds a train/test split and test-set metrics."
                ),
            )
            lr_enable_split = purpose == "Prediction / validation"

            if lr_enable_split:
                sp1, sp2 = st.columns(2)
                lr_test_size = sp1.slider(
                    "Test set size",
                    min_value=0.10,
                    max_value=0.50,
                    value=0.20,
                    step=0.05,
                    key="lr_test_size",
                )
                lr_random_state = int(sp2.number_input(
                    "Random seed",
                    min_value=1,
                    value=42,
                    step=1,
                    key="lr_random_state",
                ))
                st.caption(
                    "Split is optional and mainly useful for prediction. The full OLS model is still shown for interpretation and diagnostics."
                )

            st.markdown("#### Fix / improve model options")
            st.caption(
                "Use these when diagnostics suggest problems. For teaching, run the original model first, "
                "then add one fix at a time and compare the diagnostic checklist."
            )

            f1, f2 = st.columns(2)
            lr_outcome_transform = f1.selectbox(
                "Outcome transformation",
                ["None", "log(y)", "sqrt(y)"],
                index=0,
                key="lr_outcome_transform",
                help="Can help heteroscedasticity or non-normal residuals. It changes coefficient interpretation.",
            )
            numeric_predictors_for_poly = [
                c for c in predictors
                if c in mdf.columns and pd.api.types.is_numeric_dtype(mdf[c])
            ]
            lr_polynomial_terms = f2.multiselect(
                "Add squared term for non-linearity",
                numeric_predictors_for_poly,
                default=[],
                key="lr_polynomial_terms",
                help="Adds x^2. Useful when residual plots show curvature. Start with age or BMI if clinically plausible.",
            )
            lr_center_polynomial_terms = st.checkbox(
                "Center variables before squaring (recommended)",
                value=True,
                key="lr_center_polynomial_terms",
                help="Uses (x - mean(x))^2 instead of x^2. This usually reduces multicollinearity between x and x^2.",
            )

            h1, h2 = st.columns(2)
            lr_show_robust = h1.checkbox(
                "Show HC3 robust standard errors",
                value=True,
                key="lr_show_robust",
                help="Useful when Breusch-Pagan suggests heteroscedasticity.",
            )
            lr_compare_influential = h2.checkbox(
                "Compare model without influential observations",
                value=True,
                key="lr_compare_influential",
                help="Runs sensitivity analysis if Cook's Distance flags influential rows.",
            )

    if st.button("▶ Run Model", use_container_width=True, key="bm_run"):
        st.session_state["bm_run_requested"] = True

    # Keep model output visible across Streamlit reruns. This is important for
    # prediction mode, because uploading a new-data file triggers a rerun.
    if st.session_state.get("bm_run_requested", False):
        if not predictors:
            st.error("Select at least one predictor.")
            return
        try:
            if model_type == "Linear Regression":
                run_linear_regression(
                    mdf,
                    target,
                    predictors,
                    plot_template,
                    enable_prediction_split=lr_enable_split,
                    test_size=lr_test_size,
                    split_random_state=lr_random_state,
                    show_robust_hc3=lr_show_robust,
                    compare_without_influential=lr_compare_influential,
                    outcome_transform=lr_outcome_transform,
                    polynomial_terms=lr_polynomial_terms,
                    center_polynomial_terms=lr_center_polynomial_terms,
                )
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

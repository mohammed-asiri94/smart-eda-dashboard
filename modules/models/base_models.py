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
    mean_poisson_deviance,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
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




def _linear_term_expr(col, df):
    """Formula-safe term expression; categorical predictors are treated like as.factor() in R."""
    if col in df.columns and (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])):
        return f"C({quote_col(col)})"
    return quote_col(col)


def build_linear_formula_with_interactions(target, predictors, df, interaction_pairs=None):
    """Build OLS formula with optional interaction terms such as age:sex."""
    interaction_pairs = interaction_pairs or []
    parts = [_linear_term_expr(c, df) for c in predictors]
    for a, b in interaction_pairs:
        if a in df.columns and b in df.columns and a != b:
            parts.append(f"{_linear_term_expr(a, df)}:{_linear_term_expr(b, df)}")
    # remove duplicates while preserving order
    seen = set()
    unique_parts = []
    for part in parts:
        if part not in seen:
            unique_parts.append(part)
            seen.add(part)
    return f"{quote_col(target)} ~ {' + '.join(unique_parts)}"


def _get_training_numeric_ranges(model_df, original_predictors):
    """Store training ranges used to warn against extrapolation on new data."""
    ranges = {}
    for col in original_predictors:
        if col in model_df.columns and pd.api.types.is_numeric_dtype(model_df[col]):
            vals = pd.to_numeric(model_df[col], errors="coerce")
            ranges[col] = {
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
    return ranges


def _extrapolation_warnings(new_df, training_ranges):
    """Return row-level warnings when numeric new-data values are outside training ranges."""
    rows = []
    for col, rng in training_ranges.items():
        if col not in new_df.columns:
            continue
        vals = pd.to_numeric(new_df[col], errors="coerce")
        mask = (vals < rng["min"]) | (vals > rng["max"])
        for idx in vals[mask].index:
            rows.append({
                "Row": idx,
                "Variable": col,
                "New value": vals.loc[idx],
                "Training min": rng["min"],
                "Training max": rng["max"],
                "Warning": "Outside training range; prediction may be extrapolation and less reliable.",
            })
    return pd.DataFrame(rows)


def _prediction_summary_on_original_scale(result, new_df, outcome_transform="None", alpha=0.05):
    """Return mean prediction, confidence interval, and prediction interval on original outcome scale."""
    pred_res = result.get_prediction(new_df)
    sf = pred_res.summary_frame(alpha=alpha)
    out = pd.DataFrame({
        "Predicted": _inverse_outcome_transform(sf["mean"].values, outcome_transform),
        "Mean CI Lower": _inverse_outcome_transform(sf["mean_ci_lower"].values, outcome_transform),
        "Mean CI Upper": _inverse_outcome_transform(sf["mean_ci_upper"].values, outcome_transform),
        "Prediction PI Lower": _inverse_outcome_transform(sf["obs_ci_lower"].values, outcome_transform),
        "Prediction PI Upper": _inverse_outcome_transform(sf["obs_ci_upper"].values, outcome_transform),
    })
    return out


def _nested_model_comparison_section(full_result, model_df, model_target, reduced_predictors, interaction_pairs=None):
    """Compare a reduced OLS model with the current full model using an F-test and AIC/BIC."""
    if not reduced_predictors:
        return
    try:
        reduced_formula = build_linear_formula_with_interactions(
            model_target, reduced_predictors, model_df, interaction_pairs=[]
        )
        reduced_result = smf.ols(formula=reduced_formula, data=model_df).fit()
        st.markdown("### Nested Model Comparison / F-test")
        st.caption(
            "Compares a smaller reduced model with the current full model. A small p-value suggests the added terms improve model fit."
        )
        comp = sm.stats.anova_lm(reduced_result, full_result)
        st.dataframe(comp.reset_index(drop=True), use_container_width=True)
        metrics = pd.DataFrame({
            "Model": ["Reduced model", "Full model"],
            "Formula": [reduced_formula, str(full_result.model.formula)],
            "R2": [round(reduced_result.rsquared, 4), round(full_result.rsquared, 4)],
            "Adjusted R2": [round(reduced_result.rsquared_adj, 4), round(full_result.rsquared_adj, 4)],
            "AIC": [round(reduced_result.aic, 2), round(full_result.aic, 2)],
            "BIC": [round(reduced_result.bic, 2), round(full_result.bic, 2)],
        })
        st.dataframe(metrics, use_container_width=True)
    except Exception as e:
        st.warning("Nested model comparison could not be completed: " + str(e))

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
    training_ranges = _get_training_numeric_ranges(model_df, original_predictors)
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
        test_interval_tbl = _prediction_summary_on_original_scale(
            train_result, test_df, outcome_transform=outcome_transform, alpha=0.05
        )

        y_train_actual = train_df[original_target].astype(float).values
        y_test_actual = test_df[original_target].astype(float).values
        train_pred = _inverse_outcome_transform(train_pred_model_scale, outcome_transform)
        test_pred = test_interval_tbl["Predicted"].values

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
        st.caption(
            "Mean CI estimates uncertainty around the average prediction. "
            "Prediction PI estimates uncertainty for an individual new observation and is usually wider."
        )

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
        pred_tbl["Mean CI Lower"] = np.round(test_interval_tbl["Mean CI Lower"].values, 4)
        pred_tbl["Mean CI Upper"] = np.round(test_interval_tbl["Mean CI Upper"].values, 4)
        pred_tbl["Prediction PI Lower"] = np.round(test_interval_tbl["Prediction PI Lower"].values, 4)
        pred_tbl["Prediction PI Upper"] = np.round(test_interval_tbl["Prediction PI Upper"].values, 4)
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
                    warn_df = _extrapolation_warnings(one_df, training_ranges)
                    if not warn_df.empty:
                        st.warning("This input is outside the training data range for at least one numeric predictor.")
                        st.dataframe(warn_df, use_container_width=True)
                    interval_out = _prediction_summary_on_original_scale(
                        full_result, one_df, outcome_transform=outcome_transform, alpha=0.05
                    )
                    st.success(f"Predicted {original_target}: {interval_out.loc[0, 'Predicted']:.4f}")
                    st.dataframe(interval_out.round(4), use_container_width=True)
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
                    warn_df = _extrapolation_warnings(pred_input, training_ranges)
                    if not warn_df.empty:
                        st.warning(
                            "Some new-data values are outside the training ranges. "
                            "These predictions may be extrapolations and less reliable."
                        )
                        with st.expander("Show extrapolation warnings", expanded=False):
                            st.dataframe(warn_df, use_container_width=True)
                    interval_out = _prediction_summary_on_original_scale(
                        full_result, pred_input, outcome_transform=outcome_transform, alpha=0.05
                    )
                    output_df = new_df.copy()
                    output_df[f"predicted_{original_target}"] = np.round(interval_out["Predicted"].values, 4)
                    output_df["mean_ci_lower"] = np.round(interval_out["Mean CI Lower"].values, 4)
                    output_df["mean_ci_upper"] = np.round(interval_out["Mean CI Upper"].values, 4)
                    output_df["prediction_pi_lower"] = np.round(interval_out["Prediction PI Lower"].values, 4)
                    output_df["prediction_pi_upper"] = np.round(interval_out["Prediction PI Upper"].values, 4)
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
    interaction_pairs=None,
    compare_reduced_model=False,
    reduced_model_predictors=None,
):
    polynomial_terms = polynomial_terms or []
    interaction_pairs = interaction_pairs or []
    reduced_model_predictors = reduced_model_predictors or []
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
    formula = build_linear_formula_with_interactions(
        model_target, model_predictors, model_df, interaction_pairs=interaction_pairs
    )
    result = smf.ols(formula=formula, data=model_df).fit()

    st.success("Model fitted successfully.")
    st.code(formula, language="text")
    if outcome_transform != "None" or polynomial_notes or interaction_pairs:
        msg = transform_note
        if polynomial_notes:
            msg += " " + " | ".join(polynomial_notes)
            msg += " These terms are used to handle possible non-linearity."
        if interaction_pairs:
            interaction_text = ", ".join([f"{a} × {b}" for a, b in interaction_pairs])
            msg += f" Interaction term(s) added: {interaction_text}."
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

    if compare_reduced_model and reduced_model_predictors:
        _nested_model_comparison_section(
            result, model_df, model_target, reduced_model_predictors, interaction_pairs=interaction_pairs
        )

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
# GLM link-scale non-linearity checks (Logistic + Poisson)
# ============================================================

def _glm_bic(result):
    """Return a likelihood-based BIC when available."""
    for attr in ("bic_llf", "bic"):
        try:
            val = getattr(result, attr)
            if np.isfinite(val):
                return float(val)
        except Exception:
            pass
    return np.nan


def _glm_numeric_candidates(df, predictors, min_unique=6):
    """Numeric predictors suitable for non-linearity checks. Binary variables are skipped."""
    out = []
    for col in predictors:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        unique_vals = vals.nunique()
        if unique_vals < min_unique:
            continue
        # Skip binary/indicator-like numeric variables; transformation is not meaningful.
        uniq = set(vals.dropna().unique().tolist())
        if unique_vals <= 2 or uniq.issubset({0, 1, 0.0, 1.0}):
            continue
        out.append(col)
    return out


def _formula_term_for_col(col, df):
    """Formula-safe term for numeric or categorical columns."""
    if col in df.columns and (
        pd.api.types.is_object_dtype(df[col])
        or pd.api.types.is_categorical_dtype(df[col])
        or pd.api.types.is_bool_dtype(df[col])
    ):
        return f"C({quote_col(col)})"
    return quote_col(col)


def _build_formula_with_custom_terms(target, predictors, df, custom_terms=None):
    """Build formula while replacing selected original predictors with custom formula terms."""
    custom_terms = custom_terms or {}
    parts = []
    for col in predictors:
        if col in custom_terms:
            terms = custom_terms[col]
            if isinstance(terms, str):
                terms = [terms]
            parts.extend(terms)
        else:
            parts.append(_formula_term_for_col(col, df))
    # remove duplicate terms while preserving order
    seen = set()
    uniq = []
    for term in parts:
        if term not in seen:
            uniq.append(term)
            seen.add(term)
    return f"{quote_col(target)} ~ {' + '.join(uniq)}"


def _candidate_strength(aic_improvement):
    """Translate AIC improvement into practical recommendation strength."""
    if not np.isfinite(aic_improvement) or aic_improvement < 2:
        return "None", "Keep linear", "No meaningful improvement over the original linear term."
    if aic_improvement < 5:
        return "Weak", "Optional", "Small AIC improvement. Consider only if clinically/statistically plausible."
    if aic_improvement < 10:
        return "Moderate", "Consider", "Moderate AIC improvement. Consider this transformation and compare interpretation."
    return "Strong", "Recommended", "Large AIC improvement. This suggests meaningful non-linearity on the link scale."


def _box_tidwell_for_logistic(model_df, target, predictors, variable):
    """Box-Tidwell test for linearity of a positive continuous predictor in the logit."""
    try:
        vals = pd.to_numeric(model_df[variable], errors="coerce")
        if (vals <= 0).any():
            return np.nan, "Skipped: requires all values > 0"
        bt_df = model_df.copy()
        bt_col = _safe_internal_name("bt", variable, set(bt_df.columns))
        bt_df[bt_col] = vals * np.log(vals)
        formula = _build_formula_with_custom_terms(
            target,
            predictors,
            bt_df,
            custom_terms={variable: [quote_col(variable), quote_col(bt_col)]},
        )
        bt_res = smf.logit(formula=formula, data=bt_df).fit(disp=False)
        pval = float(bt_res.pvalues.get(quote_col(bt_col), np.nan))
        return pval, "Possible non-linearity" if np.isfinite(pval) and pval < 0.05 else "No strong evidence"
    except Exception as e:
        return np.nan, "Failed: " + str(e)[:80]


def _fit_glm_candidate(model_type, formula, cand_df, target, offset=None):
    """Fit Logistic or Poisson candidate model."""
    if model_type == "logistic":
        return smf.logit(formula=formula, data=cand_df).fit(disp=False)
    if model_type == "poisson":
        return smf.glm(
            formula=formula,
            data=cand_df,
            family=sm.families.Poisson(),
            offset=offset,
        ).fit()
    raise ValueError("Unsupported model_type")


def _candidate_metrics(model_type, res, cand_df, target, offset=None):
    """Comparable model metrics for candidate transformations."""
    out = {
        "AIC": float(res.aic),
        "BIC": _glm_bic(res),
        "Log-Likelihood": float(getattr(res, "llf", np.nan)),
    }
    if model_type == "logistic":
        y = cand_df[target].astype(int).values
        probs = np.asarray(res.predict(cand_df), dtype=float)
        try:
            out["AUC"] = float(roc_auc_score(y, probs)) if len(np.unique(y)) == 2 else np.nan
        except Exception:
            out["AUC"] = np.nan
        try:
            out["Brier"] = float(brier_score_loss(y, probs))
        except Exception:
            out["Brier"] = np.nan
        try:
            out["Log Loss"] = float(log_loss(y, probs, labels=[0, 1]))
        except Exception:
            out["Log Loss"] = np.nan
    if model_type == "poisson":
        out["Pearson χ²/df"] = float(res.pearson_chi2 / res.df_resid) if res.df_resid else np.nan
        out["Deviance/df"] = float(res.deviance / res.df_resid) if res.df_resid else np.nan
    return out


def _build_transformation_candidates(model_df, variable):
    """Create transformed data and custom formula terms for one numeric variable."""
    vals = pd.to_numeric(model_df[variable], errors="coerce")
    candidates = []

    # Original linear term
    candidates.append({
        "Form": "Linear",
        "Display": variable,
        "df": model_df.copy(),
        "custom_terms": {},
        "note": "Original linear term on the link scale.",
    })

    # Centered squared term: keep x + (x-mean)^2
    sq_df = model_df.copy()
    sq_col = _safe_internal_name("sq", variable, set(sq_df.columns))
    mean_val = float(vals.mean())
    sq_df[sq_col] = (vals - mean_val) ** 2
    candidates.append({
        "Form": "Squared",
        "Display": f"{variable} + {variable}²",
        "df": sq_df,
        "custom_terms": {variable: [quote_col(variable), quote_col(sq_col)]},
        "note": f"Uses centered squared term: ({variable} - {mean_val:.4f})².",
    })

    # Log transformation: log(x) if positive, log1p(x) if zero/non-negative.
    if (vals > 0).all():
        log_df = model_df.copy()
        log_col = _safe_internal_name("log", variable, set(log_df.columns))
        log_df[log_col] = np.log(vals)
        candidates.append({
            "Form": "Log",
            "Display": f"log({variable})",
            "df": log_df,
            "custom_terms": {variable: [quote_col(log_col)]},
            "note": "Uses natural log because all values are > 0.",
        })
    elif (vals >= 0).all():
        log_df = model_df.copy()
        log_col = _safe_internal_name("log1p", variable, set(log_df.columns))
        log_df[log_col] = np.log1p(vals)
        candidates.append({
            "Form": "Log1p",
            "Display": f"log1p({variable})",
            "df": log_df,
            "custom_terms": {variable: [quote_col(log_col)]},
            "note": "Uses log1p(x) because the variable contains zero values.",
        })

    # Square-root transformation
    if (vals >= 0).all():
        sqrt_df = model_df.copy()
        sqrt_col = _safe_internal_name("sqrt", variable, set(sqrt_df.columns))
        sqrt_df[sqrt_col] = np.sqrt(vals)
        candidates.append({
            "Form": "Sqrt",
            "Display": f"sqrt({variable})",
            "df": sqrt_df,
            "custom_terms": {variable: [quote_col(sqrt_col)]},
            "note": "Uses square-root transformation because all values are ≥ 0.",
        })

    return candidates



def _available_manual_transform_options(df, variable):
    """Available manual predictor transformations based on the data values."""
    vals = pd.to_numeric(df[variable], errors="coerce").dropna()
    opts = ["None"]
    if vals.empty:
        return opts
    opts.append("Squared: x + centered x²")
    if (vals > 0).all():
        opts.append("Log: log(x)")
    elif (vals >= 0).all():
        opts.append("Log1p: log(1+x)")
    if (vals >= 0).all():
        opts.append("Sqrt: sqrt(x)")
    return opts


def _apply_manual_predictor_transformations(model_df, predictors, transform_map=None):
    """
    Apply user-selected predictor transformations for GLM models.

    These transformations affect the fitted formula. The original predictor names remain
    in the UI and prediction inputs; internal generated columns are added to model_df.
    """
    transform_map = transform_map or {}
    df_out = model_df.copy()
    custom_terms = {}
    specs = []
    notes = []

    for col in predictors:
        choice = transform_map.get(col, "None")
        if not choice or choice == "None" or col not in df_out.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df_out[col]):
            continue
        vals = pd.to_numeric(df_out[col], errors="coerce")
        existing = set(df_out.columns)

        if choice == "Squared: x + centered x²":
            mean_val = float(vals.mean())
            new_col = _safe_internal_name("manual_sq", col, existing)
            df_out[new_col] = (vals - mean_val) ** 2
            custom_terms[col] = [quote_col(col), quote_col(new_col)]
            specs.append({"source": col, "new_col": new_col, "type": "squared_centered", "mean": mean_val})
            notes.append(f"{col}: using {col} + centered {col}² [({col} - {mean_val:.4f})²]")

        elif choice == "Log: log(x)":
            if (vals <= 0).any():
                raise ValueError(f"log({col}) requires all values to be greater than 0.")
            new_col = _safe_internal_name("manual_log", col, existing)
            df_out[new_col] = np.log(vals)
            custom_terms[col] = [quote_col(new_col)]
            specs.append({"source": col, "new_col": new_col, "type": "log"})
            notes.append(f"{col}: using log({col})")

        elif choice == "Log1p: log(1+x)":
            if (vals < 0).any():
                raise ValueError(f"log1p({col}) requires all values to be 0 or greater.")
            new_col = _safe_internal_name("manual_log1p", col, existing)
            df_out[new_col] = np.log1p(vals)
            custom_terms[col] = [quote_col(new_col)]
            specs.append({"source": col, "new_col": new_col, "type": "log1p"})
            notes.append(f"{col}: using log1p({col}) because zeros may be present")

        elif choice == "Sqrt: sqrt(x)":
            if (vals < 0).any():
                raise ValueError(f"sqrt({col}) requires all values to be 0 or greater.")
            new_col = _safe_internal_name("manual_sqrt", col, existing)
            df_out[new_col] = np.sqrt(vals)
            custom_terms[col] = [quote_col(new_col)]
            specs.append({"source": col, "new_col": new_col, "type": "sqrt"})
            notes.append(f"{col}: using sqrt({col})")

    return df_out, custom_terms, specs, notes


def _add_manual_transform_columns_to_new_data(new_df, transform_specs):
    """Create the same manual transformation columns in new prediction data."""
    if not transform_specs:
        return new_df.copy()
    out = new_df.copy()
    for spec in transform_specs:
        source = spec["source"]
        new_col = spec["new_col"]
        if source not in out.columns:
            raise ValueError(f"New data is missing required predictor column: {source}")
        vals = pd.to_numeric(out[source], errors="coerce")
        typ = spec["type"]
        if typ == "squared_centered":
            out[new_col] = (vals - float(spec.get("mean", 0.0))) ** 2
        elif typ == "log":
            if (vals <= 0).any():
                raise ValueError(f"log({source}) requires all new-data values to be greater than 0.")
            out[new_col] = np.log(vals)
        elif typ == "log1p":
            if (vals < 0).any():
                raise ValueError(f"log1p({source}) requires all new-data values to be 0 or greater.")
            out[new_col] = np.log1p(vals)
        elif typ == "sqrt":
            if (vals < 0).any():
                raise ValueError(f"sqrt({source}) requires all new-data values to be 0 or greater.")
            out[new_col] = np.sqrt(vals)
    return out


def _link_scale_linearity_check(
    model_type,
    model_df,
    target,
    predictors,
    base_result,
    plot_template,
    offset_col=None,
    offset_values=None,
):
    """Recommendation-only link-scale linearity check for Logistic and Poisson models."""
    title = "Advanced: Link-scale Linearity Check"
    st.markdown(f"### {title}")
    if model_type == "logistic":
        st.caption(
            "Checks whether numeric predictors look linear with the log-odds. "
            "This is recommendation-only; it does not change your current model."
        )
    else:
        st.caption(
            "Checks whether numeric predictors look linear with log(expected count/rate). "
            "This is recommendation-only; it does not change your current model."
        )

    numeric_vars = _glm_numeric_candidates(model_df, predictors)
    if not numeric_vars:
        st.info("No suitable numeric predictors found. Binary and categorical predictors are skipped.")
        return pd.DataFrame()

    base_aic = float(base_result.aic)
    base_bic = _glm_bic(base_result)
    rows = []
    detail_rows = []

    for var in numeric_vars:
        candidates = _build_transformation_candidates(model_df, var)
        cand_results = []
        for cand in candidates:
            try:
                cand_df = cand["df"]
                # If a Poisson offset is used, align offset to candidate dataframe index.
                cand_offset = None
                if model_type == "poisson" and offset_col:
                    cand_offset = np.log(pd.to_numeric(cand_df[offset_col], errors="coerce").astype(float))
                formula = _build_formula_with_custom_terms(
                    target,
                    predictors,
                    cand_df,
                    custom_terms=cand["custom_terms"],
                )
                res = _fit_glm_candidate(model_type, formula, cand_df, target, offset=cand_offset)
                metrics = _candidate_metrics(model_type, res, cand_df, target, offset=cand_offset)
                cand_results.append({**cand, "formula": formula, "result": res, "metrics": metrics})
                detail = {
                    "Variable": var,
                    "Candidate form": cand["Form"],
                    "Displayed form": cand["Display"],
                    "AIC": round(metrics.get("AIC", np.nan), 3),
                    "ΔAIC vs linear": round(metrics.get("AIC", np.nan) - base_aic, 3),
                    "BIC": round(metrics.get("BIC", np.nan), 3) if np.isfinite(metrics.get("BIC", np.nan)) else np.nan,
                    "Note": cand["note"],
                }
                if model_type == "logistic":
                    detail.update({
                        "AUC": round(metrics.get("AUC", np.nan), 4) if np.isfinite(metrics.get("AUC", np.nan)) else np.nan,
                        "Brier": round(metrics.get("Brier", np.nan), 4) if np.isfinite(metrics.get("Brier", np.nan)) else np.nan,
                        "Log Loss": round(metrics.get("Log Loss", np.nan), 4) if np.isfinite(metrics.get("Log Loss", np.nan)) else np.nan,
                    })
                else:
                    detail.update({
                        "Pearson χ²/df": round(metrics.get("Pearson χ²/df", np.nan), 4) if np.isfinite(metrics.get("Pearson χ²/df", np.nan)) else np.nan,
                        "Deviance/df": round(metrics.get("Deviance/df", np.nan), 4) if np.isfinite(metrics.get("Deviance/df", np.nan)) else np.nan,
                    })
                detail_rows.append(detail)
            except Exception as e:
                detail_rows.append({
                    "Variable": var,
                    "Candidate form": cand["Form"],
                    "Displayed form": cand["Display"],
                    "AIC": np.nan,
                    "ΔAIC vs linear": np.nan,
                    "BIC": np.nan,
                    "Note": "Failed: " + str(e)[:120],
                })

        good = [c for c in cand_results if np.isfinite(c["metrics"].get("AIC", np.nan))]
        if not good:
            rows.append({
                "Variable": var,
                "Best form": "Not assessed",
                "AIC improvement": np.nan,
                "Evidence": "Failed",
                "Recommendation": "No recommendation",
                "Reason": "All candidate models failed.",
            })
            continue
        best = min(good, key=lambda x: x["metrics"].get("AIC", np.inf))
        improvement = base_aic - best["metrics"].get("AIC", np.nan)
        evidence, rec, reason = _candidate_strength(improvement)

        bt_p, bt_note = (np.nan, "Not applicable")
        if model_type == "logistic":
            bt_p, bt_note = _box_tidwell_for_logistic(model_df, target, predictors, var)
            if np.isfinite(bt_p) and bt_p < 0.05 and rec == "Keep linear":
                evidence = "Possible"
                rec = "Inspect"
                reason = "Box-Tidwell suggests possible non-linearity, but AIC improvement from tested forms was small."

        rows.append({
            "Variable": var,
            "Best form": best["Display"],
            "Original AIC": round(base_aic, 3),
            "Best AIC": round(best["metrics"].get("AIC", np.nan), 3),
            "AIC improvement": round(improvement, 3) if np.isfinite(improvement) else np.nan,
            "Evidence": evidence,
            "Recommendation": rec,
            "Reason": reason,
            "Box-Tidwell p": round(bt_p, 5) if np.isfinite(bt_p) else np.nan,
            "Box-Tidwell note": bt_note,
        })

    rec_df = pd.DataFrame(rows)
    detail_df = pd.DataFrame(detail_rows)

    st.markdown("#### Recommendation table")
    st.dataframe(rec_df, use_container_width=True)

    with st.expander("Show all candidate model comparisons", expanded=False):
        st.dataframe(detail_df, use_container_width=True)

    # Simple visual summary of AIC improvement by variable.
    try:
        plot_df = rec_df.copy()
        plot_df = plot_df[np.isfinite(plot_df["AIC improvement"])]
        if not plot_df.empty:
            fig = px.bar(
                plot_df,
                x="Variable",
                y="AIC improvement",
                color="Recommendation",
                title="Best transformation AIC improvement vs original model",
                template=plot_template,
            )
            fig.add_hline(y=2, line_dash="dash", annotation_text="AIC improvement = 2")
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    st.info(
        "How to use this: keep the original model if AIC improvement is <2. "
        "If improvement is ≥2, consider the suggested form, then refit and check interpretation, calibration, and diagnostics. "
        "This section does not automatically change your model."
    )
    return rec_df


# ============================================================
# Binary Logistic Regression
# ============================================================


def _logistic_event_guidance(y_true):
    """Show event prevalence guidance, including when OR may approximate RR."""
    event_rate = float(np.mean(y_true)) if len(y_true) else np.nan
    st.markdown("#### Event prevalence and OR/RR guidance")
    if np.isfinite(event_rate):
        st.metric("Event rate", f"{event_rate * 100:.1f}%")
        if event_rate < 0.10:
            st.info("Event is rare (<10%). In this setting, Odds Ratio (OR) can approximately reflect Relative Risk (RR).")
        else:
            st.warning("Event is not rare (≥10%). OR can exaggerate the Relative Risk, so interpret OR carefully.")


def _logistic_or_interpretation(tbl):
    """Plain-language interpretation of odds ratios."""
    rows = []
    for _, row in tbl.iterrows():
        term = str(row.get("Term", ""))
        if term.lower() in ("intercept", "const"):
            continue
        if "Odds Ratio" not in row:
            continue
        try:
            or_val = float(row["Odds Ratio"])
            pval = float(row.get("P-value", np.nan))
        except Exception:
            continue
        if np.isclose(or_val, 1.0, atol=0.01):
            meaning = "approximately no change in the odds"
        elif or_val > 1:
            meaning = f"higher odds of the event by about {(or_val - 1) * 100:.1f}%"
        else:
            meaning = f"lower odds of the event by about {(1 - or_val) * 100:.1f}%"
        significance = "statistically significant" if np.isfinite(pval) and pval < 0.05 else "not statistically significant"
        rows.append({
            "Term": term,
            "OR": round(or_val, 4),
            "Plain interpretation": (
                f"Holding other predictors constant, {term} is associated with {meaning}. "
                f"This association is {significance} (p={pval:.4f})."
            ),
        })
    return pd.DataFrame(rows)


def _logistic_classification_metrics(y_true, probs, threshold=0.50):
    """Return classification metrics table and confusion matrix for binary logistic regression."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= float(threshold)).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    def safe_div(a, b):
        return float(a / b) if b else np.nan

    metrics = {
        "Accuracy": accuracy_score(y_true, preds),
        "Sensitivity / Recall": safe_div(tp, tp + fn),
        "Specificity": safe_div(tn, tn + fp),
        "Precision / PPV": safe_div(tp, tp + fp),
        "NPV": safe_div(tn, tn + fn),
        "F1-score": f1_score(y_true, preds, zero_division=0),
    }
    metrics_df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Value": [round(v, 4) if np.isfinite(v) else np.nan for v in metrics.values()],
    })
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )
    pred_class = pd.Series(preds)
    return metrics_df, cm_df, preds


def _plot_logistic_roc(y_true, probs, plot_template, title_prefix=""):
    try:
        auc = roc_auc_score(y_true, probs)
        fpr, tpr, _ = roc_curve(y_true, probs)
        fig_roc = px.line(
            x=fpr,
            y=tpr,
            labels={"x": "False Positive Rate", "y": "True Positive Rate"},
            title=f"{title_prefix}ROC Curve (AUC = {auc:.4f})",
            template=plot_template,
        )
        fig_roc.add_shape(
            type="line", x0=0, y0=0, x1=1, y1=1,
            line=dict(dash="dash", color="grey"),
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        return auc
    except Exception:
        st.info("ROC curve could not be generated.")
        return np.nan


def _plot_logistic_calibration(y_true, probs, plot_template, title="Calibration Plot"):
    try:
        y_true = np.asarray(y_true).astype(float)
        probs = np.asarray(probs).astype(float)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        mean_pred, mean_true, counts = [], [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            if hi == 1:
                mask = (probs >= lo) & (probs <= hi)
            else:
                mask = (probs >= lo) & (probs < hi)
            if mask.sum() > 0:
                mean_pred.append(float(probs[mask].mean()))
                mean_true.append(float(y_true[mask].mean()))
                counts.append(int(mask.sum()))
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=mean_pred,
            y=mean_true,
            mode="markers+lines",
            text=[f"n={c}" for c in counts],
            name="Calibration",
        ))
        fig_cal.add_shape(
            type="line", x0=0, y0=0, x1=1, y1=1,
            line=dict(dash="dash", color="red"),
        )
        fig_cal.update_layout(
            title=title,
            xaxis_title="Mean predicted probability",
            yaxis_title="Observed event rate",
            template=plot_template,
        )
        st.plotly_chart(fig_cal, use_container_width=True)
    except Exception:
        st.info("Calibration plot could not be generated.")


def _hosmer_lemeshow_test(y_true, probs, g=10):
    """Hosmer-Lemeshow goodness-of-fit test; p > 0.05 is generally desirable."""
    y_true = pd.Series(np.asarray(y_true).astype(float))
    probs = pd.Series(np.asarray(probs).astype(float))
    try:
        quantiles = pd.qcut(probs, g, duplicates="drop")
        hl_df = pd.DataFrame({"obs": y_true.values, "pred": probs.values, "bin": quantiles})
        hl_tbl = hl_df.groupby("bin", observed=False).agg(
            obs_1=("obs", "sum"),
            pred_1=("pred", "sum"),
            n=("obs", "count"),
        )
        hl_tbl["obs_0"] = hl_tbl["n"] - hl_tbl["obs_1"]
        hl_tbl["pred_0"] = hl_tbl["n"] - hl_tbl["pred_1"]
        hl_tbl = hl_tbl[(hl_tbl["pred_1"] > 0) & (hl_tbl["pred_0"] > 0)]
        if len(hl_tbl) < 3:
            return np.nan, np.nan
        hl_stat = (
            ((hl_tbl["obs_1"] - hl_tbl["pred_1"]) ** 2 / hl_tbl["pred_1"]).sum()
            + ((hl_tbl["obs_0"] - hl_tbl["pred_0"]) ** 2 / hl_tbl["pred_0"]).sum()
        )
        df_hl = max(len(hl_tbl) - 2, 1)
        hl_pval = 1 - stats.chi2.cdf(hl_stat, df=df_hl)
        return float(hl_stat), float(hl_pval)
    except Exception:
        return np.nan, np.nan


def _logistic_vif_table(result):
    """VIF after formula encoding, including categorical dummy variables."""
    exog = pd.DataFrame(result.model.exog, columns=result.model.exog_names)
    exog = exog.loc[:, exog.nunique(dropna=True) > 1]
    exog_no_intercept = exog.drop(columns=["Intercept"], errors="ignore")
    if exog_no_intercept.shape[1] < 2:
        return pd.DataFrame()
    X_vif = sm.add_constant(exog_no_intercept, has_constant="add")
    rows = []
    for i, col in enumerate(X_vif.columns):
        if col == "const":
            continue
        try:
            val = variance_inflation_factor(X_vif.values, i)
        except Exception:
            val = np.nan
        rows.append({"Variable": col, "VIF": round(float(val), 3) if np.isfinite(val) else np.nan})
    return pd.DataFrame(rows)


def _logistic_predict_on_new_data(model_df, target, predictors, result, threshold, transform_specs=None):
    st.markdown("### Predict on New Data")
    st.caption(
        "Use the fitted full-data logistic model to predict event probability for new observations. "
        "New data must contain the same predictor columns."
    )

    with st.expander("Manual single prediction", expanded=False):
        manual_values = {}
        cols = st.columns(2)
        for i, pred in enumerate(predictors):
            with cols[i % 2]:
                if pd.api.types.is_numeric_dtype(model_df[pred]):
                    default_val = float(pd.to_numeric(model_df[pred], errors="coerce").median())
                    manual_values[pred] = st.number_input(
                        pred,
                        value=default_val,
                        key=f"log_manual_pred_{pred}",
                    )
                else:
                    vals = model_df[pred].dropna().astype(str).unique().tolist()
                    vals = sorted(vals) if vals else [""]
                    manual_values[pred] = st.selectbox(
                        pred,
                        vals,
                        key=f"log_manual_pred_{pred}",
                    )
        if st.button("Predict single row", key="log_predict_single", use_container_width=True):
            try:
                one_df = pd.DataFrame([manual_values])
                for pred in predictors:
                    if not pd.api.types.is_numeric_dtype(model_df[pred]):
                        one_df[pred] = one_df[pred].astype(str)
                one_df = _add_manual_transform_columns_to_new_data(one_df, transform_specs or [])
                pred_out = result.predict(one_df)
                prob = float(pred_out.iloc[0] if hasattr(pred_out, "iloc") else pred_out[0])
                pred_class = int(prob >= threshold)
                st.success(f"Predicted probability of event: {prob:.4f} | Predicted class at threshold {threshold:.2f}: {pred_class}")
            except Exception as e:
                st.error("Prediction failed: " + str(e))

    uploaded_new = st.file_uploader(
        "Upload new data for logistic prediction (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        key="log_new_prediction_file",
    )
    if uploaded_new is not None:
        try:
            if uploaded_new.name.lower().endswith(".csv"):
                new_df = pd.read_csv(uploaded_new)
            else:
                new_df = pd.read_excel(uploaded_new)
            missing_cols = [c for c in predictors if c not in new_df.columns]
            if missing_cols:
                st.error("New data is missing required columns: " + ", ".join(missing_cols))
            else:
                pred_input = new_df.copy()
                pred_input = _add_manual_transform_columns_to_new_data(pred_input, transform_specs or [])
                probs_new = np.asarray(result.predict(pred_input), dtype=float)
                output_df = new_df.copy()
                output_df[f"predicted_probability_{target}"] = np.round(probs_new, 4)
                output_df[f"predicted_class_{target}"] = (probs_new >= threshold).astype(int)
                st.dataframe(output_df.head(50), use_container_width=True)
                st.download_button(
                    "Download logistic predictions (CSV)",
                    data=output_df.to_csv(index=False).encode(),
                    file_name=f"logistic_predictions_{target}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.error("New-data prediction failed: " + str(e))


def _logistic_prediction_workflow(model_df, target, predictors, formula, full_result, test_size, random_state, threshold, plot_template, transform_specs=None):
    st.markdown("## Logistic Prediction / Validation Workflow")
    st.info(
        "Prediction mode evaluates probability and classification performance on a held-out test set. "
        "Use the threshold slider to choose the probability cutoff for classifying event vs no event."
    )
    if len(model_df) < 30:
        st.warning("Train/test split may be unstable with small datasets. Use at least 30 observations, preferably more.")
        return
    try:
        train_df, test_df = train_test_split(
            model_df,
            test_size=float(test_size),
            random_state=int(random_state),
            stratify=model_df[target] if model_df[target].nunique() == 2 else None,
        )
        train_result = smf.logit(formula=formula, data=train_df).fit(disp=False)
        train_probs = np.asarray(train_result.predict(train_df), dtype=float)
        test_probs = np.asarray(train_result.predict(test_df), dtype=float)
        y_train = train_df[target].astype(int).values
        y_test = test_df[target].astype(int).values

        train_auc = roc_auc_score(y_train, train_probs) if len(np.unique(y_train)) == 2 else np.nan
        test_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) == 2 else np.nan
        brier = brier_score_loss(y_test, test_probs)
        ll = log_loss(y_test, test_probs, labels=[0, 1])
        metrics_df, cm_df, test_classes = _logistic_classification_metrics(y_test, test_probs, threshold)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Train N", len(train_df))
        c2.metric("Test N", len(test_df))
        c3.metric("Test AUC", round(test_auc, 4) if np.isfinite(test_auc) else "NA")
        c4.metric("Brier Score", round(brier, 4))
        c5.metric("Log Loss", round(ll, 4))

        st.markdown("### Classification metrics on test set")
        st.caption(f"Classification threshold = {threshold:.2f}. Lower threshold usually increases sensitivity; higher threshold usually increases specificity.")
        st.dataframe(metrics_df, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                aspect="auto",
                title="Test-set Confusion Matrix",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        with col_b:
            _plot_logistic_roc(y_test, test_probs, plot_template, title_prefix="Test-set ")

        _plot_logistic_calibration(y_test, test_probs, plot_template, title="Test-set Calibration Plot")

        pred_tbl = test_df[predictors].copy()
        pred_tbl["Actual"] = y_test
        pred_tbl["Predicted probability"] = np.round(test_probs, 4)
        pred_tbl["Predicted class"] = test_classes
        pred_tbl["Correct?"] = (test_classes == y_test)
        with st.expander("Optional: Test-set prediction table and download", expanded=False):
            st.dataframe(pred_tbl.head(50), use_container_width=True)
            st.download_button(
                "Download logistic test predictions (CSV)",
                data=pred_tbl.to_csv(index=False).encode(),
                file_name="logistic_test_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if np.isfinite(train_auc) and np.isfinite(test_auc) and train_auc - test_auc > 0.10:
            st.warning("Train AUC is meaningfully higher than Test AUC. This may suggest overfitting.")
        elif np.isfinite(test_auc):
            st.success("Prediction performance is evaluated on held-out test data.")

        _logistic_predict_on_new_data(model_df, target, predictors, full_result, threshold, transform_specs=transform_specs)

    except Exception as e:
        st.warning("Logistic train/test prediction workflow could not be completed: " + str(e))


def run_logistic_regression(
    df,
    target,
    predictors,
    plot_template,
    enable_prediction_split=False,
    test_size=0.20,
    split_random_state=42,
    classification_threshold=0.50,
    check_link_linearity=False,
    predictor_transform_map=None,
):
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

    manual_transform_specs = []
    try:
        model_df, manual_custom_terms, manual_transform_specs, manual_notes = _apply_manual_predictor_transformations(
            model_df, predictors, predictor_transform_map
        )
    except Exception as e:
        st.error("Manual predictor transformation failed: " + str(e))
        return

    y_true_full = model_df[target].astype(int)
    formula = _build_formula_with_custom_terms(target, predictors, model_df, custom_terms=manual_custom_terms)
    result = smf.logit(formula=formula, data=model_df).fit(disp=False)

    st.success("✅ Logistic model fitted successfully.")
    st.code(formula, language="text")
    if manual_transform_specs:
        st.info("Manual predictor transformation(s) applied: " + " | ".join(manual_notes))

    # Core model fit metrics
    mcfadden = 1 - (result.llf / result.llnull) if result.llnull != 0 else np.nan
    null_deviance = -2 * result.llnull
    residual_deviance = -2 * result.llf
    dev_reduction = (null_deviance - residual_deviance) / null_deviance * 100 if null_deviance else np.nan
    bic_val = getattr(result, "bic", np.nan)
    probs_full = np.asarray(result.predict(model_df), dtype=float)

    if enable_prediction_split:
        st.markdown("## Logistic Prediction Focus Mode")
        _logistic_prediction_workflow(
            model_df=model_df,
            target=target,
            predictors=predictors,
            formula=formula,
            full_result=result,
            test_size=test_size,
            random_state=split_random_state,
            threshold=classification_threshold,
            plot_template=plot_template,
            transform_specs=manual_transform_specs,
        )
        if check_link_linearity:
            _link_scale_linearity_check(
                model_type="logistic",
                model_df=model_df,
                target=target,
                predictors=predictors,
                base_result=result,
                plot_template=plot_template,
            )
        with st.expander("Optional: Logistic inference summary", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("N", int(result.nobs))
            c2.metric("McFadden R²", round(mcfadden, 4) if np.isfinite(mcfadden) else "NA")
            c3.metric("AIC", round(result.aic, 2))
            c4.metric("BIC", round(bic_val, 2) if np.isfinite(bic_val) else "NA")
            tbl = coef_table(result, "Binary Logistic Regression")
            st.dataframe(tbl, use_container_width=True)
            interp = _logistic_or_interpretation(tbl)
            if not interp.empty:
                st.dataframe(interp, use_container_width=True)
        return

    # Inference / explanation mode
    st.markdown("### Model Fit Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("N", int(result.nobs))
    c2.metric("McFadden R²", round(mcfadden, 4) if np.isfinite(mcfadden) else "NA")
    c3.metric("AIC", round(result.aic, 2))
    c4.metric("BIC", round(bic_val, 2) if np.isfinite(bic_val) else "NA")
    c5.metric("Log-Likelihood", round(result.llf, 2))

    st.markdown("#### Null vs Residual Deviance")
    dev_df = pd.DataFrame({
        "Metric": ["Null Deviance", "Residual Deviance", "Deviance reduction %", "Likelihood ratio p-value"],
        "Value": [
            round(null_deviance, 4),
            round(residual_deviance, 4),
            round(dev_reduction, 2) if np.isfinite(dev_reduction) else np.nan,
            round(float(result.llr_pvalue), 6) if hasattr(result, "llr_pvalue") else np.nan,
        ],
        "Interpretation": [
            "Model with intercept only.",
            "Full model; lower than null deviance is better.",
            "How much deviance was reduced by adding predictors.",
            "Tests whether predictors improve fit over intercept-only model.",
        ],
    })
    st.dataframe(dev_df, use_container_width=True)

    # Coefficients with OR
    st.markdown("### Coefficients + Odds Ratios")
    tbl = coef_table(result, "Binary Logistic Regression")
    st.dataframe(tbl, use_container_width=True)
    download_buttons(result, tbl, "Binary Logistic Regression")

    interp = _logistic_or_interpretation(tbl)
    if not interp.empty:
        with st.expander("Plain-language OR interpretation", expanded=True):
            st.dataframe(interp, use_container_width=True)

    _logistic_event_guidance(y_true_full)

    # Full-data probability and classification evaluation
    st.markdown("### Probability and Classification Performance")
    try:
        auc = roc_auc_score(y_true_full, probs_full)
    except Exception:
        auc = np.nan
    try:
        brier = brier_score_loss(y_true_full, probs_full)
    except Exception:
        brier = np.nan
    try:
        ll = log_loss(y_true_full, probs_full, labels=[0, 1])
    except Exception:
        ll = np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("AUC", round(auc, 4) if np.isfinite(auc) else "NA")
    c2.metric("Brier Score", round(brier, 4) if np.isfinite(brier) else "NA")
    c3.metric("Log Loss", round(ll, 4) if np.isfinite(ll) else "NA")
    st.caption("AUC higher is better. Brier Score and Log Loss lower are better.")

    st.markdown("#### Classification at selected threshold")
    st.caption(f"Current threshold = {classification_threshold:.2f}. Change it in Logistic Regression options.")
    metrics_df, cm_df, preds_class = _logistic_classification_metrics(y_true_full, probs_full, classification_threshold)
    st.dataframe(metrics_df, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    with col_b:
        _plot_logistic_roc(y_true_full, probs_full, plot_template)

    _plot_logistic_calibration(y_true_full, probs_full, plot_template)

    # Diagnostics
    st.markdown("### 🔬 Diagnostics")
    issues = []

    class_counts = y_true_full.value_counts().sort_index()
    minority_pct = class_counts.min() / len(y_true_full) * 100
    st.markdown(
        f"**Class balance:** {class_counts.to_dict()} "
        f"(minority class = {minority_pct:.1f}%)"
    )
    if minority_pct < 20:
        issues.append({
            "level": "warning",
            "msg": f"Class imbalance — minority class is only {minority_pct:.1f}% of data.",
            "fix": "Consider threshold tuning, stratified split, resampling, or class-weighted models for prediction.",
        })

    hl_stat, hl_pval = _hosmer_lemeshow_test(y_true_full, probs_full, g=10)
    if np.isfinite(hl_pval):
        st.markdown(f"**Hosmer-Lemeshow test** — χ² = `{hl_stat:.4f}`, p-value = `{hl_pval:.4f}` (good calibration if p > 0.05)")
        if hl_pval < 0.05:
            issues.append({
                "level": "warning",
                "msg": f"Hosmer-Lemeshow suggests poor calibration (p = {hl_pval:.4f}).",
                "fix": "Add non-linear terms, interactions, more predictors, or consider calibration methods.",
            })
    else:
        st.info("Hosmer-Lemeshow test could not be calculated.")

    try:
        vif_data = _logistic_vif_table(result)
        if not vif_data.empty:
            st.markdown("#### Multicollinearity: VIF")
            st.caption("VIF is calculated after formula encoding, so categorical predictors are checked through dummy variables.")
            st.dataframe(vif_data, use_container_width=True)
            high_vif = vif_data[vif_data["VIF"] > 10]
            if not high_vif.empty:
                issues.append({
                    "level": "error",
                    "msg": f"Multicollinearity detected (VIF > 10): {', '.join(high_vif['Variable'].astype(str).tolist())}",
                    "fix": "Remove or combine highly correlated predictors.",
                })
    except Exception:
        pass

    if mcfadden < 0.1:
        issues.append({
            "level": "warning",
            "msg": f"McFadden R² = {mcfadden:.4f} — model has low explanatory power.",
            "fix": "Consider adding more relevant predictors, non-linear terms, or interaction terms.",
        })

    if check_link_linearity:
        _link_scale_linearity_check(
            model_type="logistic",
            model_df=model_df,
            target=target,
            predictors=predictors,
            base_result=result,
            plot_template=plot_template,
        )

    _logistic_predict_on_new_data(model_df, target, predictors, result, classification_threshold, transform_specs=manual_transform_specs)

    st.markdown("### 🩺 Diagnostic Summary")
    show_diagnostics_header(issues)

    with st.expander("Full Model Summary"):
        st.text(result.summary().as_text())


# ============================================================
# Poisson Regression
# ============================================================

def _poisson_irr_interpretation(tbl):
    """Plain-language interpretation for incidence rate ratios."""
    rows = []
    if "IRR" not in tbl.columns:
        return pd.DataFrame(rows)
    for _, row in tbl.iterrows():
        term = str(row.get("Term", ""))
        if term.lower() in ("intercept", "const"):
            continue
        irr = float(row.get("IRR", np.nan))
        pval = float(row.get("P-value", np.nan))
        if not np.isfinite(irr):
            continue
        if irr > 1:
            pct = (irr - 1) * 100
            meaning = f"higher event rate by about {pct:.1f}%"
        elif irr < 1:
            pct = (1 - irr) * 100
            meaning = f"lower event rate by about {pct:.1f}%"
        else:
            meaning = "no change in event rate"
        significance = "statistically significant" if np.isfinite(pval) and pval < 0.05 else "not statistically significant"
        rows.append({
            "Term": term,
            "IRR": round(irr, 4),
            "Plain interpretation": (
                f"Holding other predictors constant, this term is associated with {meaning}. "
                f"The association is {significance}."
            ),
        })
    return pd.DataFrame(rows)


def _poisson_extrapolation_warnings(train_df, new_df, predictors):
    """Return warnings when new numeric values are outside the training range."""
    warnings = []
    for col in predictors:
        if col in train_df.columns and col in new_df.columns and pd.api.types.is_numeric_dtype(train_df[col]):
            train_vals = pd.to_numeric(train_df[col], errors="coerce")
            new_vals = pd.to_numeric(new_df[col], errors="coerce")
            lo = float(train_vals.min())
            hi = float(train_vals.max())
            below = int((new_vals < lo).sum())
            above = int((new_vals > hi).sum())
            if below or above:
                warnings.append(f"{col}: {below} value(s) below training min {lo:.4f}, {above} above training max {hi:.4f}")
    return warnings


def _poisson_predict_on_new_data(model_df, target, predictors, result, exposure_col=None, transform_specs=None):
    """Manual and file-based prediction for Poisson models."""
    st.markdown("### Predict on New Data")
    st.caption(
        "New data must contain the same predictor columns used in the model. "
        "If an exposure/offset was used, the new data must also include that exposure column."
    )

    required_cols = list(predictors) + ([exposure_col] if exposure_col else [])

    with st.expander("Manual single prediction", expanded=False):
        manual_values = {}
        cols = st.columns(2)
        for i, pred in enumerate(required_cols):
            with cols[i % 2]:
                if pred == exposure_col:
                    default_val = float(pd.to_numeric(model_df[pred], errors="coerce").median())
                    manual_values[pred] = st.number_input(
                        f"{pred} (exposure)",
                        min_value=0.000001,
                        value=max(default_val, 0.000001),
                        key=f"pois_manual_exposure_{pred}",
                    )
                elif pd.api.types.is_numeric_dtype(model_df[pred]):
                    default_val = float(pd.to_numeric(model_df[pred], errors="coerce").median())
                    manual_values[pred] = st.number_input(pred, value=default_val, key=f"pois_manual_{pred}")
                else:
                    vals = model_df[pred].dropna().astype(str).unique().tolist()
                    vals = sorted(vals) if vals else [""]
                    manual_values[pred] = st.selectbox(pred, vals, key=f"pois_manual_{pred}")
        if st.button("Predict count", key="pois_predict_single", use_container_width=True):
            try:
                one_df = pd.DataFrame([manual_values])
                for pred in predictors:
                    if not pd.api.types.is_numeric_dtype(model_df[pred]):
                        one_df[pred] = one_df[pred].astype(str)
                offset = None
                if exposure_col:
                    if (one_df[exposure_col] <= 0).any():
                        st.error("Exposure values must be greater than 0.")
                        return
                    offset = np.log(one_df[exposure_col].astype(float))
                warn = _poisson_extrapolation_warnings(model_df, one_df, predictors)
                if warn:
                    st.warning("Extrapolation warning: " + " | ".join(warn))
                one_df = _add_manual_transform_columns_to_new_data(one_df, transform_specs or [])
                pred_count = float(result.predict(one_df, offset=offset).iloc[0])
                st.success(f"Predicted {target}: {pred_count:.4f}")
                if exposure_col:
                    exposure_value = float(one_df[exposure_col].iloc[0])
                    st.info(f"Predicted rate per exposure unit: {pred_count / exposure_value:.4f}")
            except Exception as e:
                st.error("Prediction failed: " + str(e))

    uploaded_new = st.file_uploader(
        "Upload new data for Poisson prediction (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        key="pois_new_prediction_file",
    )
    if uploaded_new is not None:
        try:
            if uploaded_new.name.lower().endswith(".csv"):
                new_df = pd.read_csv(uploaded_new)
            else:
                new_df = pd.read_excel(uploaded_new)
            missing_cols = [c for c in required_cols if c not in new_df.columns]
            if missing_cols:
                st.error("New data is missing required columns: " + ", ".join(missing_cols))
                return
            pred_input = new_df.copy()
            offset = None
            if exposure_col:
                if (pd.to_numeric(pred_input[exposure_col], errors="coerce") <= 0).any():
                    st.error("Exposure values must be greater than 0 in the uploaded file.")
                    return
                offset = np.log(pd.to_numeric(pred_input[exposure_col], errors="coerce"))
            warn = _poisson_extrapolation_warnings(model_df, pred_input, predictors)
            if warn:
                st.warning("Extrapolation warning: " + " | ".join(warn))
            pred_input = _add_manual_transform_columns_to_new_data(pred_input, transform_specs or [])
            pred_counts = result.predict(pred_input, offset=offset)
            output_df = new_df.copy()
            output_df[f"predicted_{target}"] = np.round(pred_counts, 4)
            if exposure_col:
                output_df[f"predicted_rate_per_{exposure_col}"] = np.round(pred_counts / pd.to_numeric(output_df[exposure_col], errors="coerce"), 4)
            st.dataframe(output_df.head(50), use_container_width=True)
            st.download_button(
                "Download Poisson predictions (CSV)",
                data=output_df.to_csv(index=False).encode(),
                file_name=f"poisson_predicted_{target}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error("New-data prediction failed: " + str(e))


def _poisson_prediction_section(model_df, target, predictors, formula, full_result, test_size, random_state, plot_template, exposure_col=None):
    """Train/test validation for Poisson count prediction."""
    st.markdown("## Poisson Prediction / Validation Workflow")
    st.info(
        "Prediction mode trains the Poisson model on a training set and evaluates count prediction on a held-out test set."
    )
    if len(model_df) < 20:
        st.warning("Train/test split is not reliable with very small datasets. Use at least 20 observations, preferably more.")
        return
    try:
        train_df, test_df = train_test_split(model_df, test_size=float(test_size), random_state=int(random_state))
        train_offset = np.log(train_df[exposure_col].astype(float)) if exposure_col else None
        test_offset = np.log(test_df[exposure_col].astype(float)) if exposure_col else None
        train_result = smf.glm(
            formula=formula,
            data=train_df,
            family=sm.families.Poisson(),
            offset=train_offset,
        ).fit()
        y_train = train_df[target].astype(float).values
        y_test = test_df[target].astype(float).values
        train_pred = np.asarray(train_result.predict(train_df, offset=train_offset), dtype=float)
        test_pred = np.asarray(train_result.predict(test_df, offset=test_offset), dtype=float)
        train_pred = np.clip(train_pred, 1e-12, None)
        test_pred = np.clip(test_pred, 1e-12, None)
        train_mae = float(mean_absolute_error(y_train, train_pred))
        test_mae = float(mean_absolute_error(y_test, test_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
        mpd = float(mean_poisson_deviance(y_test, test_pred))
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Train N", len(train_df))
        c2.metric("Test N", len(test_df))
        c3.metric("Test MAE", round(test_mae, 4))
        c4.metric("Test RMSE", round(rmse, 4))
        c5.metric("Mean Poisson Deviance", round(mpd, 4))
        st.caption("For MAE, RMSE, and Mean Poisson Deviance, lower is better.")
        if test_mae > train_mae * 1.5 and len(train_df) > 0:
            st.warning("Test MAE is much higher than Train MAE. This may indicate unstable prediction or overfitting.")
        else:
            st.success("Train and test count prediction errors are reasonably consistent.")

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.scatter(
                x=y_test, y=test_pred,
                labels={"x": "Actual test counts", "y": "Predicted test counts"},
                title="Test Set: Actual vs Predicted Count",
                template=plot_template,
            )
            max_v = max(float(np.nanmax(y_test)), float(np.nanmax(test_pred)))
            fig.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v, line=dict(dash="dash"))
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            error = y_test - test_pred
            fig_err = px.scatter(
                x=test_pred, y=error,
                labels={"x": "Predicted test counts", "y": "Prediction error"},
                title="Test Set: Prediction Error vs Predicted",
                template=plot_template,
            )
            fig_err.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig_err, use_container_width=True)

        pred_tbl = test_df[predictors + ([exposure_col] if exposure_col else [])].copy()
        pred_tbl["Actual"] = np.round(y_test, 4)
        pred_tbl["Predicted"] = np.round(test_pred, 4)
        pred_tbl["Error"] = np.round(y_test - test_pred, 4)
        pred_tbl["Absolute Error"] = np.round(np.abs(y_test - test_pred), 4)
        with st.expander("Optional: Test-set prediction table and download", expanded=False):
            st.dataframe(pred_tbl.head(50), use_container_width=True)
            st.download_button(
                "Download Poisson test predictions (CSV)",
                data=pred_tbl.to_csv(index=False).encode(),
                file_name="poisson_test_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
    except Exception as e:
        st.warning("Poisson train/test prediction workflow could not be completed: " + str(e))


def run_poisson_regression(
    df,
    target,
    predictors,
    plot_template,
    enable_prediction_split=False,
    test_size=0.20,
    split_random_state=42,
    exposure_col=None,
    compare_negative_binomial=True,
    check_link_linearity=False,
    predictor_transform_map=None,
):
    cols = [target] + predictors + ([exposure_col] if exposure_col else [])
    model_df = df[cols].dropna().copy()
    if model_df.empty:
        st.error("No data remaining after dropping missing values.")
        return

    if not pd.api.types.is_numeric_dtype(model_df[target]):
        st.error("Poisson Regression requires a numeric count outcome.")
        return
    if (model_df[target] < 0).any():
        st.error("Poisson Regression requires non-negative count values.")
        return
    if not np.allclose(model_df[target], np.round(model_df[target])):
        st.warning("Poisson outcomes are usually integer counts. Your target has non-integer values; check whether Poisson is appropriate.")

    offset = None
    if exposure_col:
        if not pd.api.types.is_numeric_dtype(model_df[exposure_col]):
            st.error("Exposure / offset variable must be numeric.")
            return
        if (model_df[exposure_col] <= 0).any():
            st.error("Exposure / offset variable must be greater than 0.")
            return
        offset = np.log(model_df[exposure_col].astype(float))
        st.info(f"Using exposure offset: offset(log({exposure_col})). The model estimates event rates adjusted for exposure.")

    manual_transform_specs = []
    try:
        model_df, manual_custom_terms, manual_transform_specs, manual_notes = _apply_manual_predictor_transformations(
            model_df, predictors, predictor_transform_map
        )
    except Exception as e:
        st.error("Manual predictor transformation failed: " + str(e))
        return

    mean_t = model_df[target].mean()
    var_t = model_df[target].var()
    var_mean_ratio = var_t / mean_t if mean_t > 0 else np.nan

    formula = _build_formula_with_custom_terms(target, predictors, model_df, custom_terms=manual_custom_terms)
    result = smf.glm(
        formula=formula,
        data=model_df,
        family=sm.families.Poisson(),
        offset=offset,
    ).fit()

    st.success("✅ Poisson model fitted successfully.")
    st.code(formula + (f" + offset(log({exposure_col}))" if exposure_col else ""), language="text")
    if manual_transform_specs:
        st.info("Manual predictor transformation(s) applied: " + " | ".join(manual_notes))

    pearson_disp = float(result.pearson_chi2 / result.df_resid) if result.df_resid else np.nan
    deviance_disp = float(result.deviance / result.df_resid) if result.df_resid else np.nan
    bic_val = getattr(result, "bic_llf", np.nan)
    if not np.isfinite(bic_val):
        try:
            bic_val = result.bic
        except Exception:
            bic_val = np.nan

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("N", int(result.nobs))
    c2.metric("Mean count", round(mean_t, 3))
    c3.metric("Variance", round(var_t, 3))
    c4.metric("Var/Mean", round(var_mean_ratio, 3) if np.isfinite(var_mean_ratio) else "NA")
    c5.metric("AIC", round(result.aic, 2))

    c6, c7, c8, c9 = st.columns(4)
    c6.metric("BIC", round(bic_val, 2) if np.isfinite(bic_val) else "NA")
    c7.metric("Deviance", round(result.deviance, 2))
    c8.metric("Pearson χ²/df", round(pearson_disp, 3) if np.isfinite(pearson_disp) else "NA")
    c9.metric("Deviance/df", round(deviance_disp, 3) if np.isfinite(deviance_disp) else "NA")

    st.markdown("### Coefficients + Incidence Rate Ratios (IRR)")
    tbl = coef_table(result, "Poisson Regression")
    st.dataframe(tbl, use_container_width=True)
    download_buttons(result, tbl, "Poisson Regression")

    irr_interp = _poisson_irr_interpretation(tbl)
    if not irr_interp.empty:
        with st.expander("Plain-language IRR interpretation", expanded=True):
            st.dataframe(irr_interp, use_container_width=True)

    preds = np.asarray(result.predict(model_df, offset=offset), dtype=float)
    preds = np.clip(preds, 1e-12, None)

    if enable_prediction_split:
        st.markdown("## Poisson Prediction Focus Mode")
        _poisson_prediction_section(
            model_df=model_df,
            target=target,
            predictors=predictors,
            formula=formula,
            full_result=result,
            test_size=test_size,
            random_state=split_random_state,
            plot_template=plot_template,
            exposure_col=exposure_col,
        )
        _poisson_predict_on_new_data(model_df, target, predictors, result, exposure_col=exposure_col, transform_specs=manual_transform_specs)
        if check_link_linearity:
            _link_scale_linearity_check(
                model_type="poisson",
                model_df=model_df,
                target=target,
                predictors=predictors,
                base_result=result,
                plot_template=plot_template,
                offset_col=exposure_col,
            )
        show_full_poisson_diagnostics = st.checkbox(
            "Show full Poisson diagnostics in prediction mode",
            value=False,
            key="pois_show_full_diagnostics",
        )
        if not show_full_poisson_diagnostics:
            st.info("Full Poisson diagnostics are hidden in prediction mode. Enable the checkbox above to inspect them.")
            return

    st.markdown("### Model Fit Plot")
    fig = px.scatter(
        x=model_df[target], y=preds,
        labels={"x": "Actual Count", "y": "Predicted Count"},
        title="Actual vs Predicted Count",
        template=plot_template,
    )
    max_v = max(float(model_df[target].max()), float(np.max(preds)))
    fig.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v, line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔬 Poisson Diagnostics")
    issues = []

    dispersion_df = pd.DataFrame({
        "Check": ["Variance/Mean", "Pearson χ²/df", "Deviance/df"],
        "Value": [
            round(var_mean_ratio, 4) if np.isfinite(var_mean_ratio) else np.nan,
            round(pearson_disp, 4) if np.isfinite(pearson_disp) else np.nan,
            round(deviance_disp, 4) if np.isfinite(deviance_disp) else np.nan,
        ],
        "Interpretation": [
            "Poisson expects mean approximately equal to variance.",
            "Values >1.5 suggest overdispersion; >2 is strong.",
            "Values >1.5 also suggest overdispersion.",
        ],
    })
    st.dataframe(dispersion_df, use_container_width=True)

    if np.isfinite(pearson_disp) and pearson_disp > 2:
        issues.append({
            "level": "error",
            "msg": f"Strong overdispersion detected (Pearson χ²/df = {pearson_disp:.3f}).",
            "fix": "Use Negative Binomial Regression or another count model that allows overdispersion.",
        })
    elif np.isfinite(pearson_disp) and pearson_disp > 1.5:
        issues.append({
            "level": "warning",
            "msg": f"Overdispersion detected (Pearson χ²/df = {pearson_disp:.3f}).",
            "fix": "Compare with Negative Binomial Regression and consider robust standard errors.",
        })
    elif np.isfinite(pearson_disp) and pearson_disp > 1.2:
        issues.append({
            "level": "info",
            "msg": f"Mild overdispersion possible (Pearson χ²/df = {pearson_disp:.3f}).",
            "fix": "Inspect fit and compare AIC with Negative Binomial if needed.",
        })

    zero_pct = (model_df[target] == 0).mean() * 100
    expected_zero_pct = np.exp(-mean_t) * 100 if mean_t >= 0 else np.nan
    zero_ratio = zero_pct / expected_zero_pct if expected_zero_pct and expected_zero_pct > 0 else np.nan
    zero_df = pd.DataFrame({
        "Metric": ["Observed zero %", "Expected zero % under Poisson", "Observed / expected zero ratio"],
        "Value": [
            round(zero_pct, 2),
            round(expected_zero_pct, 2) if np.isfinite(expected_zero_pct) else np.nan,
            round(zero_ratio, 3) if np.isfinite(zero_ratio) else np.nan,
        ],
    })
    st.markdown("#### Zero inflation check")
    st.dataframe(zero_df, use_container_width=True)
    if np.isfinite(zero_ratio) and zero_ratio > 1.5:
        issues.append({
            "level": "warning",
            "msg": f"Excess zeros detected ({zero_pct:.1f}% observed vs {expected_zero_pct:.1f}% expected).",
            "fix": "Consider Zero-Inflated Poisson or Zero-Inflated Negative Binomial if zeros have a separate data-generating process.",
        })

    pearson_resid = np.asarray(result.resid_pearson)
    dev_resid = np.asarray(result.resid_deviance)
    col_a, col_b = st.columns(2)
    with col_a:
        fig_pr = px.scatter(
            x=preds, y=pearson_resid,
            labels={"x": "Predicted count", "y": "Pearson residual"},
            title="Pearson Residuals vs Predicted",
            template=plot_template,
        )
        fig_pr.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_pr, use_container_width=True)
    with col_b:
        fig_dr = px.scatter(
            x=preds, y=dev_resid,
            labels={"x": "Predicted count", "y": "Deviance residual"},
            title="Deviance Residuals vs Predicted",
            template=plot_template,
        )
        fig_dr.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_dr, use_container_width=True)

    if check_link_linearity:
        _link_scale_linearity_check(
            model_type="poisson",
            model_df=model_df,
            target=target,
            predictors=predictors,
            base_result=result,
            plot_template=plot_template,
            offset_col=exposure_col,
        )

    if compare_negative_binomial:
        st.markdown("### Poisson vs Negative Binomial Comparison")
        try:
            nb_result = smf.glm(
                formula=formula,
                data=model_df,
                family=sm.families.NegativeBinomial(),
                offset=offset,
            ).fit()
            nb_bic = getattr(nb_result, "bic_llf", np.nan)
            if not np.isfinite(nb_bic):
                try:
                    nb_bic = nb_result.bic
                except Exception:
                    nb_bic = np.nan
            comp_df = pd.DataFrame({
                "Metric": ["AIC", "BIC", "Pearson χ²/df"],
                "Poisson": [
                    round(result.aic, 2),
                    round(bic_val, 2) if np.isfinite(bic_val) else np.nan,
                    round(pearson_disp, 4) if np.isfinite(pearson_disp) else np.nan,
                ],
                "Negative Binomial": [
                    round(nb_result.aic, 2),
                    round(nb_bic, 2) if np.isfinite(nb_bic) else np.nan,
                    round(float(nb_result.pearson_chi2 / nb_result.df_resid), 4) if nb_result.df_resid else np.nan,
                ],
            })
            st.dataframe(comp_df, use_container_width=True)
            if nb_result.aic + 2 < result.aic:
                st.warning("Negative Binomial has meaningfully lower AIC. It may fit better, especially if overdispersion is present.")
            else:
                st.info("Poisson AIC is similar to or lower than Negative Binomial. Poisson may be adequate if diagnostics are acceptable.")
        except Exception as e:
            st.info("Negative Binomial comparison could not be completed: " + str(e))

    if not enable_prediction_split:
        _poisson_predict_on_new_data(model_df, target, predictors, result, exposure_col=exposure_col, transform_specs=manual_transform_specs)

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
    st.caption("Base Models version: v10_glm_manual_predictor_transformations")
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
    lr_interaction_pairs = []
    lr_compare_reduced_model = False
    lr_reduced_model_predictors = []

    log_enable_split = False
    log_test_size = 0.20
    log_random_state = 42
    log_threshold = 0.50
    log_check_link_linearity = False
    log_predictor_transform_map = {}

    pois_enable_split = False
    pois_test_size = 0.20
    pois_random_state = 42
    pois_exposure_col = None
    pois_compare_nb = True
    pois_check_link_linearity = False
    pois_predictor_transform_map = {}

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

            st.markdown("#### Interaction terms")
            st.caption("Use interactions when the effect of one predictor may depend on another predictor, e.g. age × sex.")
            possible_interactions = []
            for i, a in enumerate(predictors):
                for b in predictors[i + 1:]:
                    possible_interactions.append(f"{a} × {b}")
            selected_interactions = st.multiselect(
                "Add interaction term(s)",
                possible_interactions,
                default=[],
                key="lr_interaction_terms",
                help="Adds interaction-only terms to the OLS formula. Main effects remain in the model.",
            )
            lr_interaction_pairs = []
            for item in selected_interactions:
                if " × " in item:
                    a, b = item.split(" × ", 1)
                    lr_interaction_pairs.append((a, b))

            if not lr_enable_split:
                st.markdown("#### Nested model comparison / F-test")
                lr_compare_reduced_model = st.checkbox(
                    "Compare with a reduced model",
                    value=False,
                    key="lr_compare_reduced_model",
                    help="Fits a smaller model and compares it with the current full model using an F-test plus AIC/BIC.",
                )
                if lr_compare_reduced_model:
                    lr_reduced_model_predictors = st.multiselect(
                        "Reduced model predictors",
                        predictors,
                        default=predictors[:max(1, min(2, len(predictors)))],
                        key="lr_reduced_model_predictors",
                        help="Choose the predictors for the smaller model. The current model is treated as the full model.",
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

    if model_type == "Binary Logistic Regression":
        with st.expander("Logistic Regression options", expanded=True):
            log_purpose = st.radio(
                "Analysis purpose",
                [
                    "Statistical inference / explanation",
                    "Prediction / validation",
                ],
                horizontal=True,
                key="log_purpose",
                help=(
                    "Inference focuses on odds ratios, p-values, calibration, and model fit. "
                    "Prediction adds train/test validation, threshold tuning, Brier Score, Log Loss, and new-data prediction."
                ),
            )
            log_enable_split = log_purpose == "Prediction / validation"
            log_threshold = st.slider(
                "Classification threshold",
                min_value=0.05,
                max_value=0.95,
                value=0.50,
                step=0.05,
                key="log_threshold",
                help="Probability cutoff for classifying event=1. Lower values usually increase sensitivity; higher values usually increase specificity.",
            )

            st.markdown("#### Advanced: Link-scale linearity")
            log_check_link_linearity = st.checkbox(
                "Check non-linearity for numeric predictors (recommendations only)",
                value=False,
                key="log_check_link_linearity",
                help=(
                    "Automatically checks numeric predictors using squared, log/log1p, and sqrt candidates. "
                    "The app only gives recommendations and does not change the fitted model."
                ),
            )

            log_apply_manual_transform = st.checkbox(
                "Apply predictor transformation manually",
                value=False,
                key="log_apply_manual_transform",
                help=(
                    "Use this after the recommendation table suggests a transformation. "
                    "This changes the fitted logistic formula and keeps new-data prediction compatible."
                ),
            )
            if log_apply_manual_transform:
                st.caption("Choose transformations for numeric predictors only. Categorical and binary variables are skipped.")
                log_numeric_transform_candidates = _glm_numeric_candidates(mdf, predictors)
                if not log_numeric_transform_candidates:
                    st.info("No suitable numeric predictors available for manual transformation.")
                for var in log_numeric_transform_candidates:
                    opts = _available_manual_transform_options(mdf, var)
                    log_predictor_transform_map[var] = st.selectbox(
                        f"Transform {var}",
                        opts,
                        index=0,
                        key=f"log_manual_transform_{var}",
                    )

            if log_enable_split:
                lp1, lp2 = st.columns(2)
                log_test_size = lp1.slider(
                    "Test set size",
                    min_value=0.10,
                    max_value=0.50,
                    value=0.20,
                    step=0.05,
                    key="log_test_size",
                )
                log_random_state = int(lp2.number_input(
                    "Random seed",
                    min_value=1,
                    value=42,
                    step=1,
                    key="log_random_state",
                ))

    if model_type == "Poisson Regression":
        with st.expander("Poisson Regression options", expanded=True):
            pois_purpose = st.radio(
                "Analysis purpose",
                [
                    "Statistical inference / explanation",
                    "Prediction / validation",
                ],
                horizontal=True,
                key="pois_purpose",
                help=(
                    "Inference focuses on IRR, overdispersion, zero inflation, and model fit. "
                    "Prediction adds train/test validation and new-data count prediction."
                ),
            )
            pois_enable_split = pois_purpose == "Prediction / validation"

            if pois_enable_split:
                pp1, pp2 = st.columns(2)
                pois_test_size = pp1.slider(
                    "Test set size",
                    min_value=0.10,
                    max_value=0.50,
                    value=0.20,
                    step=0.05,
                    key="pois_test_size",
                )
                pois_random_state = int(pp2.number_input(
                    "Random seed",
                    min_value=1,
                    value=42,
                    step=1,
                    key="pois_random_state",
                ))

            st.markdown("#### Rate model / exposure offset")
            use_exposure = st.checkbox(
                "Use exposure / offset variable",
                value=False,
                key="pois_use_exposure",
                help="Use this when counts occur over different exposure times, populations, person-years, follow-up days, or days at risk.",
            )
            if use_exposure:
                exposure_candidates = [c for c in available if c not in predictors and pd.api.types.is_numeric_dtype(mdf[c])]
                if not exposure_candidates:
                    st.warning("No numeric exposure columns available. Exposure must be numeric and greater than 0.")
                    pois_exposure_col = None
                else:
                    pois_exposure_col = st.selectbox(
                        "Exposure variable",
                        exposure_candidates,
                        key="pois_exposure_col",
                        help="The model will use offset(log(exposure)). Exposure values must be greater than 0.",
                    )

            pois_compare_nb = st.checkbox(
                "Compare with Negative Binomial",
                value=True,
                key="pois_compare_nb",
                help="Useful when overdispersion is present. Compares AIC/BIC and dispersion with a Negative Binomial model.",
            )

            st.markdown("#### Advanced: Link-scale linearity")
            pois_check_link_linearity = st.checkbox(
                "Check non-linearity for numeric predictors (recommendations only)",
                value=False,
                key="pois_check_link_linearity",
                help=(
                    "Automatically checks numeric predictors using squared, log/log1p, and sqrt candidates. "
                    "The app only gives recommendations and does not change the fitted model."
                ),
            )

            pois_apply_manual_transform = st.checkbox(
                "Apply predictor transformation manually",
                value=False,
                key="pois_apply_manual_transform",
                help=(
                    "Use this after the recommendation table suggests a transformation. "
                    "This changes the fitted Poisson formula and keeps new-data prediction compatible."
                ),
            )
            if pois_apply_manual_transform:
                st.caption("Choose transformations for numeric predictors only. Categorical and binary variables are skipped.")
                pois_numeric_transform_candidates = _glm_numeric_candidates(mdf, predictors)
                if not pois_numeric_transform_candidates:
                    st.info("No suitable numeric predictors available for manual transformation.")
                for var in pois_numeric_transform_candidates:
                    opts = _available_manual_transform_options(mdf, var)
                    pois_predictor_transform_map[var] = st.selectbox(
                        f"Transform {var}",
                        opts,
                        index=0,
                        key=f"pois_manual_transform_{var}",
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
                    interaction_pairs=lr_interaction_pairs,
                    compare_reduced_model=lr_compare_reduced_model,
                    reduced_model_predictors=lr_reduced_model_predictors,
                )
            elif model_type == "Binary Logistic Regression":
                run_logistic_regression(
                    mdf,
                    target,
                    predictors,
                    plot_template,
                    enable_prediction_split=log_enable_split,
                    test_size=log_test_size,
                    split_random_state=log_random_state,
                    classification_threshold=log_threshold,
                    check_link_linearity=log_check_link_linearity,
                    predictor_transform_map=log_predictor_transform_map,
                )
            elif model_type == "Poisson Regression":
                run_poisson_regression(
                    mdf,
                    target,
                    predictors,
                    plot_template,
                    enable_prediction_split=pois_enable_split,
                    test_size=pois_test_size,
                    split_random_state=pois_random_state,
                    exposure_col=pois_exposure_col,
                    compare_negative_binomial=pois_compare_nb,
                    check_link_linearity=pois_check_link_linearity,
                    predictor_transform_map=pois_predictor_transform_map,
                )
            elif model_type == "Negative Binomial Regression":
                run_negative_binomial(mdf, target, predictors, plot_template)
        except Exception as e:
            st.error(f"Model error: {e}")
            with st.expander("Error details"):
                import traceback
                st.code(traceback.format_exc())

# ============================================================
# modules/models/time_series.py
# Time Series Analysis + Interrupted Time Series (ITS)
# Full diagnostics + fix suggestions on the same page
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# statsmodels
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm


# ============================================================
# Shared helpers
# ============================================================

def _show_issue(level, msg, fix=""):
    if level == "error":
        st.error(f"🔴 **{msg}**" + (f"\n\n💡 *Fix:* {fix}" if fix else ""))
    elif level == "warning":
        st.warning(f"🟡 **{msg}**" + (f"\n\n💡 *Fix:* {fix}" if fix else ""))
    else:
        st.info(f"🔵 {msg}")


def _diagnostic_summary(issues):
    if not issues:
        st.success("✅ All diagnostic checks passed.")
    else:
        for i in issues:
            _show_issue(i["level"], i["msg"], i.get("fix", ""))


# ============================================================
# Stationarity tests
# ============================================================

def _run_adf(series):
    """Augmented Dickey-Fuller test — H0: unit root / non-stationary."""
    result = adfuller(series.dropna(), autolag="AIC")

    return {
        "stat": round(result[0], 4),
        "pval": round(result[1], 4),
        "lags": result[2],
        "stationary": result[1] < 0.05,
    }


def _run_kpss(series):
    """KPSS test — H0: stationary."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = kpss(series.dropna(), regression="c", nlags="auto")

    return {
        "stat": round(result[0], 4),
        "pval": round(result[1], 4),
        "stationary": result[1] > 0.05,
    }


def _show_stationarity(series):
    """Run ADF + KPSS and show results."""
    issues = []
    st.markdown("#### Stationarity Tests")

    clean_series = series.dropna()

    if len(clean_series) < 8:
        issues.append({
            "level": "warning",
            "msg": "Not enough observations for reliable stationarity testing.",
            "fix": "Use at least 8 observations, preferably more, before running ADF/KPSS tests.",
        })
        _diagnostic_summary(issues)
        return issues, False

    try:
        adf = _run_adf(clean_series)
        kpss_r = _run_kpss(clean_series)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**ADF Test** (H₀: non-stationary)")
            st.markdown(
                f"Statistic: `{adf['stat']}` | p-value: `{adf['pval']}`\n\n"
                + (
                    "✅ Stationary (p < 0.05)"
                    if adf["stationary"]
                    else "🔴 Non-stationary (p ≥ 0.05)"
                )
            )

        with col2:
            st.markdown("**KPSS Test** (H₀: stationary)")
            st.markdown(
                f"Statistic: `{kpss_r['stat']}` | p-value: `{kpss_r['pval']}`\n\n"
                + (
                    "✅ Stationary (p > 0.05)"
                    if kpss_r["stationary"]
                    else "🔴 Non-stationary (p ≤ 0.05)"
                )
            )

        if not adf["stationary"] and not kpss_r["stationary"]:
            issues.append({
                "level": "error",
                "msg": "Both ADF and KPSS indicate non-stationarity.",
                "fix": (
                    "Apply first differencing (d=1) or log-transform before modeling. "
                    "Re-test after differencing."
                ),
            })

        elif adf["stationary"] and not kpss_r["stationary"]:
            issues.append({
                "level": "warning",
                "msg": "Mixed stationarity signals — possible trend-stationary series.",
                "fix": "Consider detrending instead of differencing.",
            })

        elif not adf["stationary"] and kpss_r["stationary"]:
            issues.append({
                "level": "warning",
                "msg": "Mixed signals — possibly difference-stationary.",
                "fix": "Apply one round of differencing (d=1) and retest.",
            })

        return issues, adf["stationary"]

    except Exception as e:
        issues.append({
            "level": "warning",
            "msg": f"Stationarity tests could not be completed: {e}",
            "fix": "Check that the selected series is numeric and has enough non-missing observations.",
        })
        _diagnostic_summary(issues)
        return issues, False


# ============================================================
# Time Series Analysis + ARIMA
# ============================================================

def run_time_series(series, freq, plot_template):
    st.markdown("## 📈 Time Series Analysis")

    series = series.dropna()
    issues = []
    n = len(series)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", n)
    c2.metric("Mean", round(series.mean(), 3))
    c3.metric("Std Dev", round(series.std(), 3))
    c4.metric("Missing", int(series.isnull().sum()))

    if n < 24:
        issues.append({
            "level": "warning",
            "msg": f"Only {n} observations — time series analysis requires more data for reliable results.",
            "fix": "Collect more time points. A minimum of 24 observations is recommended.",
        })

    # ========================================================
    # Raw series plot
    # ========================================================
    st.markdown("### Raw Series")

    fig_raw = go.Figure()
    fig_raw.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name="Observed",
            line=dict(color="#2563EB", width=1.5),
        )
    )

    fig_raw.update_layout(
        title="Time Series Plot",
        xaxis_title="Time",
        yaxis_title="Value",
        template=plot_template,
    )

    st.plotly_chart(fig_raw, use_container_width=True)

    # ========================================================
    # Decomposition
    # ========================================================
    st.markdown("### Decomposition (Trend + Seasonality + Residuals)")

    min_periods = freq * 2 if freq else 4

    if n >= min_periods:
        try:
            decomp_model = st.radio(
                "Decomposition model",
                ["additive", "multiplicative"],
                horizontal=True,
                key="ts_decomp_model",
            )

            if decomp_model == "multiplicative" and (series <= 0).any():
                st.warning(
                    "Multiplicative decomposition requires all positive values. "
                    "Switching to additive."
                )
                decomp_model = "additive"

            period = freq if freq and freq >= 2 else 2

            decomp = seasonal_decompose(
                series.dropna(),
                model=decomp_model,
                period=period
            )

            fig_decomp = go.Figure()

            components = [
                ("Observed", series.dropna().values, "#2563EB"),
                ("Trend", decomp.trend.values, "#16A34A"),
                ("Seasonal", decomp.seasonal.values, "#F59E0B"),
                ("Residual", decomp.resid.values, "#DC2626"),
            ]

            decomp_index = series.dropna().index

            for i, (name, vals, color) in enumerate(components):
                fig_decomp.add_trace(
                    go.Scatter(
                        x=decomp_index,
                        y=vals,
                        mode="lines",
                        name=name,
                        line=dict(color=color, width=1.5),
                        visible=True if i == 0 else "legendonly",
                    )
                )

            fig_decomp.update_layout(
                title=f"Decomposition ({decomp_model})",
                xaxis_title="Time",
                yaxis_title="Value",
                template=plot_template,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )

            st.plotly_chart(fig_decomp, use_container_width=True)
            st.caption("Click legend items to show or hide components.")

        except Exception as e:
            st.warning(f"Decomposition failed: {e}")
    else:
        st.info(
            f"Need at least {min_periods} observations for decomposition with period={freq}."
        )

    # ========================================================
    # Stationarity
    # ========================================================
    st.markdown("### Stationarity Tests")
    stat_issues, is_stationary = _show_stationarity(series)
    issues.extend(stat_issues)

    # ========================================================
    # ACF / PACF
    # ========================================================
    st.markdown("### ACF and PACF")

    max_lags = min(40, n // 2 - 1)

    if max_lags >= 2:
        try:
            clean_series = series.dropna()

            acf_vals = acf(clean_series, nlags=max_lags, fft=True)
            pacf_vals = pacf(clean_series, nlags=max_lags)
            conf_int = 1.96 / np.sqrt(len(clean_series))

            col_acf, col_pacf = st.columns(2)

            with col_acf:
                fig_acf = go.Figure()
                lags = list(range(len(acf_vals)))

                colors_acf = [
                    "#DC2626" if abs(v) > conf_int else "#2563EB"
                    for v in acf_vals
                ]

                fig_acf.add_trace(
                    go.Bar(
                        x=lags,
                        y=acf_vals,
                        marker_color=colors_acf,
                        name="ACF",
                    )
                )

                fig_acf.add_hline(y=conf_int, line_dash="dash", line_color="grey")
                fig_acf.add_hline(y=-conf_int, line_dash="dash", line_color="grey")

                fig_acf.update_layout(
                    title="ACF",
                    xaxis_title="Lag",
                    yaxis_title="Autocorrelation",
                    template=plot_template,
                )

                st.plotly_chart(fig_acf, use_container_width=True)

            with col_pacf:
                fig_pacf = go.Figure()

                colors_pacf = [
                    "#DC2626" if abs(v) > conf_int else "#2563EB"
                    for v in pacf_vals
                ]

                fig_pacf.add_trace(
                    go.Bar(
                        x=list(range(len(pacf_vals))),
                        y=pacf_vals,
                        marker_color=colors_pacf,
                        name="PACF",
                    )
                )

                fig_pacf.add_hline(y=conf_int, line_dash="dash", line_color="grey")
                fig_pacf.add_hline(y=-conf_int, line_dash="dash", line_color="grey")

                fig_pacf.update_layout(
                    title="PACF",
                    xaxis_title="Lag",
                    yaxis_title="Partial Autocorrelation",
                    template=plot_template,
                )

                st.plotly_chart(fig_pacf, use_container_width=True)

            st.caption(
                "Red bars exceed the 95% confidence interval — significant autocorrelation at those lags."
            )

        except Exception as e:
            st.warning(f"ACF/PACF could not be computed: {e}")
    else:
        st.info("Not enough observations to compute ACF/PACF.")

    # ========================================================
    # ARIMA Forecast
    # ========================================================
    st.markdown("### ARIMA Forecast")

    st.info(
        "Select ARIMA parameters manually or use auto-detection. "
        "ACF/PACF above can help guide the choice of p and q."
    )

    fc1, fc2, fc3, fc4 = st.columns(4)
    use_auto = fc1.checkbox("Auto ARIMA", value=False, key="ts_auto_arima")

    if use_auto:
        try:
            import pmdarima as pm

            with st.spinner("Running Auto ARIMA..."):
                auto_model = pm.auto_arima(
                    series.dropna(),
                    seasonal=True,
                    m=freq if freq else 1,
                    suppress_warnings=True,
                    error_action="ignore",
                    stepwise=True,
                )

            p, d, q = auto_model.order
            P, D, Q, m = auto_model.seasonal_order

            st.success(
                f"Auto ARIMA selected: ARIMA({p},{d},{q}) × ({P},{D},{Q})[{m}]"
            )

        except ImportError:
            st.warning("pmdarima not installed. Using manual parameters.")
            use_auto = False
            p, d, q = 1, 1, 1

        except Exception as e:
            st.warning(f"Auto ARIMA failed: {e}. Using manual parameters.")
            use_auto = False
            p, d, q = 1, 1, 1

    else:
        p = fc2.number_input("p (AR)", 0, 5, 1, key="ts_p")
        d = fc3.number_input("d (I)", 0, 2, 1, key="ts_d")
        q = fc4.number_input("q (MA)", 0, 5, 1, key="ts_q")

    forecast_steps = st.slider(
        "Forecast steps ahead",
        1,
        min(n, 36),
        min(12, max(1, n // 4))
    )

    if "arima_results" not in st.session_state:
        st.session_state["arima_results"] = None

    col_btn1, col_btn2 = st.columns([3, 1])

    with col_btn1:
        run_arima = st.button(
            "▶ Fit ARIMA and Forecast",
            key="ts_arima_btn",
            use_container_width=True
        )

    with col_btn2:
        if st.button("🗑 Clear", key="ts_arima_clear", use_container_width=True):
            st.session_state["arima_results"] = None

    if run_arima:
        try:
            with st.spinner("Fitting ARIMA model..."):
                model = SARIMAX(
                    series.dropna(),
                    order=(int(p), int(d), int(q)),
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )

                fitted = model.fit(disp=False)

            forecast_obj = fitted.get_forecast(steps=forecast_steps)
            forecast_mean = forecast_obj.predicted_mean
            conf_int_fc = forecast_obj.conf_int()
            resid = fitted.resid.dropna()
            dw = durbin_watson(resid)

            if len(resid) <= 5000 and len(resid) >= 3:
                _, sw_p = stats.shapiro(resid)
            else:
                sw_p = None

            st.session_state["arima_results"] = {
                "p": int(p),
                "d": int(d),
                "q": int(q),
                "aic": round(fitted.aic, 2),
                "bic": round(fitted.bic, 2),
                "llf": round(fitted.llf, 2),
                "series_index": list(series.index),
                "series_values": list(series.values),
                "forecast_index": list(forecast_mean.index),
                "forecast_values": list(forecast_mean.values),
                "ci_lower": list(conf_int_fc.iloc[:, 0].values),
                "ci_upper": list(conf_int_fc.iloc[:, 1].values),
                "dw": round(dw, 4),
                "sw_p": round(sw_p, 4) if sw_p else None,
                "resid": list(resid.values),
                "summary": fitted.summary().as_text(),
            }

        except Exception as e:
            st.error(f"ARIMA fitting failed: {e}")
            with st.expander("Error details"):
                import traceback
                st.code(traceback.format_exc())

    res = st.session_state.get("arima_results")

    if res:
        fig_fc = go.Figure()

        fig_fc.add_trace(
            go.Scatter(
                x=res["series_index"],
                y=res["series_values"],
                mode="lines",
                name="Observed",
                line=dict(color="#2563EB", width=1.5),
            )
        )

        fig_fc.add_trace(
            go.Scatter(
                x=res["forecast_index"],
                y=res["forecast_values"],
                mode="lines",
                name="Forecast",
                line=dict(color="#DC2626", width=2, dash="dash"),
            )
        )

        fig_fc.add_trace(
            go.Scatter(
                x=res["forecast_index"] + res["forecast_index"][::-1],
                y=res["ci_upper"] + res["ci_lower"][::-1],
                fill="toself",
                fillcolor="rgba(220,38,38,0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
                showlegend=True,
            )
        )

        fig_fc.update_layout(
            title=f"ARIMA({res['p']},{res['d']},{res['q']}) Forecast",
            xaxis_title="Time",
            yaxis_title="Value",
            template=plot_template,
        )

        st.plotly_chart(fig_fc, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("AIC", res["aic"])
        m2.metric("BIC", res["bic"])
        m3.metric("Log-L", res["llf"])

        st.markdown("#### Residual Diagnostics")

        rd1, rd2 = st.columns(2)

        rd1.metric(
            "Durbin-Watson",
            res["dw"],
            help="Ideal value is approximately 2.0. Values far from 2 suggest autocorrelation.",
        )

        rd2.metric(
            "Shapiro-Wilk p",
            res["sw_p"] if res["sw_p"] else "N/A",
            help="p > 0.05 means residuals are approximately normal.",
        )

        arima_issues = []

        if res["dw"] < 1.5 or res["dw"] > 2.5:
            arima_issues.append({
                "level": "warning",
                "msg": f"Residual autocorrelation detected (DW = {res['dw']}).",
                "fix": "Increase p or q, or add seasonal components such as SARIMA.",
            })

        if res["sw_p"] and res["sw_p"] < 0.05:
            arima_issues.append({
                "level": "warning",
                "msg": "Residuals may not be normally distributed.",
                "fix": "Consider log-transforming the series.",
            })

        _diagnostic_summary(arima_issues)

        with st.expander("Forecast Table"):
            fc_tbl = pd.DataFrame({
                "Time": res["forecast_index"],
                "Forecast": [round(v, 4) for v in res["forecast_values"]],
                "Lower 95%": [round(v, 4) for v in res["ci_lower"]],
                "Upper 95%": [round(v, 4) for v in res["ci_upper"]],
            })

            st.dataframe(fc_tbl, use_container_width=True)

            st.download_button(
                "📥 Download forecast (CSV)",
                data=fc_tbl.to_csv(index=False).encode(),
                file_name="arima_forecast.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.expander("Full ARIMA Summary"):
            st.text(res["summary"])

    if issues:
        st.markdown("### 🩺 Diagnostic Summary")
        _diagnostic_summary(issues)


# ============================================================
# Interrupted Time Series (ITS)
# ============================================================

def run_its(series, intervention_point, plot_template, control_series=None):
    st.markdown("## 📉📈 Interrupted Time Series (ITS)")

    st.info(
        "ITS evaluates the **causal effect of an intervention** on a time series. "
        "It estimates: (1) immediate **level change** and (2) **slope change** after the intervention."
    )

    series = series.dropna()
    n = len(series)
    issues = []

    if n < 8:
        st.error("Need at least 8 observations to run ITS.")
        return

    time_var = np.arange(1, n + 1)
    index_list = list(series.index)

    try:
        intervention_idx = index_list.index(intervention_point)
    except ValueError:
        try:
            intervention_idx = min(
                range(len(index_list)),
                key=lambda i: abs(pd.Timestamp(index_list[i]) - pd.Timestamp(intervention_point))
            )
        except Exception:
            st.error("Could not locate the selected intervention point in the time index.")
            return

    intervention_flag = (time_var > intervention_idx).astype(int)

    time_after = np.where(
        time_var > intervention_idx,
        time_var - intervention_idx,
        0,
    )

    pre_n = intervention_idx
    post_n = n - intervention_idx

    c1, c2, c3 = st.columns(3)
    c1.metric("Pre-intervention periods", pre_n)
    c2.metric("Post-intervention periods", post_n)
    c3.metric("Total observations", n)

    if pre_n < 8:
        issues.append({
            "level": "error",
            "msg": f"Only {pre_n} pre-intervention periods — very low statistical power.",
            "fix": "At least 8–12 pre-intervention periods are recommended for reliable estimates.",
        })

    if post_n < 4:
        issues.append({
            "level": "warning",
            "msg": f"Only {post_n} post-intervention periods — estimates may be unstable.",
            "fix": "Extend follow-up period for more reliable estimates.",
        })

    X = sm.add_constant(
        np.column_stack([time_var, intervention_flag, time_after])
    )

    X_df = pd.DataFrame(
        X,
        columns=["const", "Time", "Intervention", "Time_After"],
        index=series.index,
    )

    try:
        model = sm.OLS(series.values, X_df)
        result = model.fit(
            cov_type="HAC",
            cov_kwds={"maxlags": max(1, min(4, pre_n // 2))}
        )
    except Exception as e:
        st.error(f"ITS model fitting failed: {e}")
        with st.expander("Error details"):
            import traceback
            st.code(traceback.format_exc())
        return

    st.markdown("### ITS Model Coefficients")

    st.caption(
        "β₁ (Time): Pre-intervention trend | "
        "β₂ (Intervention): Immediate level change | "
        "β₃ (Time_After): Change in slope after intervention"
    )

    params = np.asarray(result.params).ravel()
    conf = np.asarray(result.conf_int())
    pvals = np.asarray(result.pvalues).ravel()
    bse = np.asarray(result.bse).ravel()

    if conf.ndim == 2 and conf.shape[1] >= 2:
        ci_lo = conf[:, 0]
        ci_hi = conf[:, 1]
    else:
        ci_lo = np.full(len(params), np.nan)
        ci_hi = np.full(len(params), np.nan)

    parameter_names = [
        "Intercept (β₀)",
        "Pre-trend (β₁)",
        "Level change (β₂)",
        "Slope change (β₃)"
    ]

    min_len = min(
        len(parameter_names),
        len(params),
        len(bse),
        len(pvals),
        len(ci_lo),
        len(ci_hi)
    )

    coef_tbl = pd.DataFrame({
        "Parameter": parameter_names[:min_len],
        "Coefficient": np.round(params[:min_len], 4),
        "Std Error": np.round(bse[:min_len], 4),
        "p-value": np.round(pvals[:min_len], 5),
        "CI Lower": np.round(ci_lo[:min_len], 4),
        "CI Upper": np.round(ci_hi[:min_len], 4),
        "Significant": ["✅" if p < 0.05 else "" for p in pvals[:min_len]],
    })

    st.dataframe(coef_tbl, use_container_width=True)

    st.download_button(
        "📥 Download ITS coefficients (CSV)",
        data=coef_tbl.to_csv(index=False).encode(),
        file_name="its_coefficients.csv",
        mime="text/csv",
        use_container_width=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("R²", round(result.rsquared, 4))
    m2.metric("Adj. R²", round(result.rsquared_adj, 4))
    m3.metric("F-statistic", round(result.fvalue, 3))

    st.markdown("### Interpretation")

    beta1 = float(params[1]) if len(params) > 1 else np.nan
    beta2 = float(params[2]) if len(params) > 2 else np.nan
    beta3 = float(params[3]) if len(params) > 3 else np.nan

    p1 = float(pvals[1]) if len(pvals) > 1 else np.nan
    p2 = float(pvals[2]) if len(pvals) > 2 else np.nan
    p3 = float(pvals[3]) if len(pvals) > 3 else np.nan

    direction_pre = "increasing" if beta1 > 0 else "decreasing"
    direction_lev = "increase" if beta2 > 0 else "decrease"
    direction_slp = "accelerated" if beta3 > 0 else "decelerated"

    st.markdown(f"""
| Effect | Estimate | Significant? | Interpretation |
|--------|----------|--------------|----------------|
| Pre-intervention trend (β₁) | {beta1:.4f} | {"✅" if p1 < 0.05 else "❌"} | Series was {direction_pre} by {abs(beta1):.4f} per period before intervention |
| Immediate level change (β₂) | {beta2:.4f} | {"✅" if p2 < 0.05 else "❌"} | Intervention caused an immediate {direction_lev} of {abs(beta2):.4f} |
| Slope change (β₃) | {beta3:.4f} | {"✅" if p3 < 0.05 else "❌"} | Post-intervention trend {direction_slp} by {abs(beta3):.4f} per period |
    """)

    # ========================================================
    # ITS plot with counterfactual
    # ========================================================
    st.markdown("### ITS Plot with Counterfactual")

    st.caption(
        "The **counterfactual** dashed red line shows what would have happened "
        "if the intervention had not occurred."
    )

    fitted_vals = result.fittedvalues

    X_cf = np.column_stack([
        np.ones(n),
        time_var,
        np.zeros(n),
        np.zeros(n),
    ])

    counterfactual = X_cf @ params[:4]

    fig_its = go.Figure()

    fig_its.add_vrect(
        x0=series.index[0],
        x1=intervention_point,
        fillcolor="rgba(37,99,235,0.05)",
        layer="below",
        line_width=0,
        annotation_text="Pre",
        annotation_position="top left",
    )

    fig_its.add_vrect(
        x0=intervention_point,
        x1=series.index[-1],
        fillcolor="rgba(220,38,38,0.05)",
        layer="below",
        line_width=0,
        annotation_text="Post",
        annotation_position="top right",
    )

    fig_its.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="markers+lines",
            name="Observed",
            line=dict(color="#2563EB", width=1.5),
            marker=dict(size=4),
        )
    )

    fig_its.add_trace(
        go.Scatter(
            x=series.index,
            y=fitted_vals,
            mode="lines",
            name="ITS Fitted",
            line=dict(color="#16A34A", width=2),
        )
    )

    fig_its.add_trace(
        go.Scatter(
            x=series.index,
            y=counterfactual,
            mode="lines",
            name="Counterfactual",
            line=dict(color="#DC2626", width=2, dash="dash"),
        )
    )

    # FIXED: use add_shape + add_annotation instead of add_vline
    # This avoids Plotly error with mixed date/string x-axis values.
    fig_its.add_shape(
        type="line",
        x0=intervention_point,
        x1=intervention_point,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(
            color="#F59E0B",
            width=2,
            dash="dash"
        )
    )

    fig_its.add_annotation(
        x=intervention_point,
        y=1,
        xref="x",
        yref="paper",
        text="Intervention",
        showarrow=False,
        yshift=15,
        font=dict(
            color="#F59E0B",
            size=12
        )
    )

    fig_its.update_layout(
        title="Interrupted Time Series",
        xaxis_title="Time",
        yaxis_title="Value",
        template=plot_template,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    st.plotly_chart(fig_its, use_container_width=True)

    last_observed = series.values[-1]
    last_counterfactual = counterfactual[-1]
    effect_at_end = last_observed - last_counterfactual

    st.metric(
        "Estimated effect at last time point",
        f"{effect_at_end:.4f}",
        help="Observed minus counterfactual at the last time point.",
    )

    # ========================================================
    # Controlled ITS
    # ========================================================
    if control_series is not None:
        st.markdown("### Controlled ITS — Difference-in-Differences")

        st.info(
            "Subtracting the control series removes common trends "
            "and strengthens causal inference."
        )

        control_series = control_series.reindex(series.index)
        diff_series = series - control_series
        diff_series = diff_series.dropna()

        if len(diff_series) > 10:
            try:
                diff_n = len(diff_series)
                diff_time = np.arange(1, diff_n + 1)

                diff_index_list = list(diff_series.index)

                try:
                    diff_intervention_idx = diff_index_list.index(intervention_point)
                except ValueError:
                    diff_intervention_idx = min(
                        range(len(diff_index_list)),
                        key=lambda i: abs(pd.Timestamp(diff_index_list[i]) - pd.Timestamp(intervention_point))
                    )

                X_ctrl = sm.add_constant(
                    np.column_stack([
                        diff_time,
                        np.where(diff_time > diff_intervention_idx, 1, 0),
                        np.where(
                            diff_time > diff_intervention_idx,
                            diff_time - diff_intervention_idx,
                            0
                        ),
                    ])
                )

                ctrl_model = sm.OLS(diff_series.values, X_ctrl)
                ctrl_result = ctrl_model.fit(
                    cov_type="HAC",
                    cov_kwds={"maxlags": 4}
                )

                c_params = np.asarray(ctrl_result.params).ravel()
                c_pvals = np.asarray(ctrl_result.pvalues).ravel()

                if len(c_params) >= 3 and len(c_pvals) >= 3:
                    st.markdown(
                        f"**Controlled level change (β₂):** "
                        f"`{c_params[2]:.4f}` "
                        f"({'✅ significant' if c_pvals[2] < 0.05 else '❌ not significant'}, "
                        f"p = {c_pvals[2]:.4f})"
                    )

            except Exception as e:
                st.warning(f"Controlled ITS could not be completed: {e}")
        else:
            st.info("Not enough aligned observations for controlled ITS.")

    # ========================================================
    # Diagnostics
    # ========================================================
    st.markdown("### 🔬 Diagnostics")

    dw = durbin_watson(result.resid)

    st.markdown(f"**Durbin-Watson:** `{dw:.4f}` (ideal ≈ 2.0)")

    if dw < 1.5 or dw > 2.5:
        issues.append({
            "level": "warning",
            "msg": (
                f"Autocorrelation in ITS residuals (DW = {dw:.3f}). "
                "HAC standard errors have been applied automatically."
            ),
            "fix": (
                "HAC (Newey-West) standard errors are already used. "
                "If severe, consider adding a lagged outcome as a covariate."
            ),
        })

    st.markdown("#### Chow Test — Structural Break at Intervention Point")

    try:
        pre_series = series.values[:intervention_idx]
        post_series = series.values[intervention_idx:]

        n_pre = len(pre_series)
        n_post = len(post_series)

        if n_pre >= 4 and n_post >= 4:
            X_pre = sm.add_constant(np.arange(1, n_pre + 1))
            X_post = sm.add_constant(np.arange(1, n_post + 1))

            ssr_pre = sm.OLS(pre_series, X_pre).fit().ssr
            ssr_post = sm.OLS(post_series, X_post).fit().ssr
            ssr_full = result.ssr

            k = X.shape[1]
            denom_df = n - 2 * k

            if denom_df > 0:
                chow_f = ((ssr_full - ssr_pre - ssr_post) / k) / (
                    (ssr_pre + ssr_post) / denom_df
                )

                chow_p = 1 - stats.f.cdf(chow_f, k, denom_df)

                st.markdown(
                    f"Chow F-statistic: `{chow_f:.4f}` | p-value: `{chow_p:.4f}`"
                )

                if chow_p < 0.05:
                    st.success(
                        "✅ Structural break confirmed at intervention point (p < 0.05)."
                    )
                else:
                    issues.append({
                        "level": "warning",
                        "msg": (
                            f"Chow test does not confirm structural break "
                            f"(p = {chow_p:.4f})."
                        ),
                        "fix": (
                            "The intervention may not have caused a detectable change. "
                            "Verify the intervention date or consider a longer post period."
                        ),
                    })
            else:
                st.info("Not enough degrees of freedom for Chow test.")
        else:
            st.info("Not enough data in pre/post segments for Chow test.")

    except Exception as e:
        st.info(f"Chow test could not be performed: {e}")

    try:
        stat_issues, _ = _show_stationarity(
            pd.Series(result.resid, index=series.index)
        )

        if stat_issues:
            issues.append({
                "level": "warning",
                "msg": "ITS residuals may be non-stationary.",
                "fix": "Consider adding seasonal dummies or applying differencing to the outcome before ITS.",
            })

    except Exception as e:
        st.info(f"Residual stationarity check could not be completed: {e}")

    st.markdown("### 🩺 Diagnostic Summary")
    _diagnostic_summary(issues)

    with st.expander("Full ITS Model Summary"):
        st.text(result.summary().as_text())


# ============================================================
# Main render function — called from app.py
# ============================================================

def render_time_series_tab(df, df_cleaned, plot_template):
    st.markdown("# 📅 Time Series Analysis")

    dataset_choice = st.radio(
        "Dataset to use",
        ["Original data", "Cleaned data (from Data Cleaning tab)"],
        horizontal=True,
        key="ts_dataset",
    )

    mdf = df_cleaned.copy() if dataset_choice.startswith("Cleaned") else df.copy()

    all_cols = mdf.columns.tolist()
    numeric_cols = mdf.select_dtypes(include=np.number).columns.tolist()

    if len(all_cols) == 0 or len(numeric_cols) == 0:
        st.warning("The dataset must contain at least one time column and one numeric value column.")
        return

    st.markdown("### Column Setup")

    ts1, ts2, ts3 = st.columns(3)

    with ts1:
        time_col = st.selectbox(
            "🗓 Time / Date column",
            all_cols,
            key="ts_time_col",
            help="Will be used as the time index. Can be a date or a sequential number.",
        )

    with ts2:
        value_options = [c for c in numeric_cols if c != time_col]

        if not value_options:
            st.warning("Please select a dataset with at least one numeric value column.")
            return

        value_col = st.selectbox(
            "📊 Value column (outcome)",
            value_options,
            key="ts_value_col",
        )

    with ts3:
        freq_options = {
            "Auto-detect": None,
            "Daily (7)": 7,
            "Monthly (12)": 12,
            "Quarterly (4)": 4,
            "Weekly (52)": 52,
            "Annual (1)": 1,
            "Custom": -1,
        }

        freq_label = st.selectbox(
            "🔄 Seasonality period",
            list(freq_options.keys()),
            key="ts_freq",
        )

        freq = freq_options[freq_label]

        if freq == -1:
            freq = st.number_input(
                "Custom period",
                2,
                365,
                12,
                key="ts_custom_freq"
            )

    st.markdown("### Analysis Type")

    analysis_type = st.radio(
        "Choose analysis",
        [
            "📈 Time Series Analysis + ARIMA Forecast",
            "📉📈 Interrupted Time Series (ITS)"
        ],
        horizontal=True,
        key="ts_analysis_type",
    )

    intervention_point = None
    control_col = None

    if "ITS" in analysis_type:
        st.markdown("### ITS Settings")
        its1, its2 = st.columns(2)

        with its1:
            st.markdown("**Intervention point**")
            st.caption(
                "Select the time point at which the intervention occurred. "
                "Must have enough observations before and after."
            )

        with its2:
            control_col_options = ["None"] + [
                c for c in numeric_cols
                if c not in (time_col, value_col)
            ]

            control_col = st.selectbox(
                "Control series (optional — for controlled ITS)",
                control_col_options,
                key="its_control",
                help=(
                    "A parallel series not affected by the intervention. "
                    "Strengthens causal inference."
                ),
            )

        with st.expander("📖 What is ITS?"):
            st.markdown("""
**Interrupted Time Series (ITS)** is a quasi-experimental design that:

- Requires a clearly defined **intervention date**
- Estimates two effects:
  - **Level change (β₂):** Did the series jump or drop immediately?
  - **Slope change (β₃):** Did the rate of change accelerate or decelerate?
- Uses **Newey-West HAC standard errors** to handle autocorrelation automatically
- **Counterfactual line:** Shows what would have happened without the intervention
- **Chow test:** Helps assess whether a structural break occurred at the intervention point

**Controlled ITS** uses a parallel control series to remove common trends.
            """)

    ts_df = mdf[[time_col, value_col]].dropna().copy()

    if ts_df.empty:
        st.warning("No usable rows after removing missing values from the selected time/value columns.")
        return

    try:
        ts_df[time_col] = pd.to_datetime(ts_df[time_col])
    except Exception:
        pass

    try:
        ts_df = ts_df.sort_values(time_col).set_index(time_col)
    except Exception:
        ts_df = ts_df.set_index(time_col)

    series_preview = ts_df[value_col]

    if "ITS" in analysis_type and len(series_preview) >= 8:
        time_index = list(series_preview.index)

        if len(time_index) > 8:
            valid_range = time_index[4:-4]
        else:
            valid_range = time_index[1:-1]

        intervention_point = st.selectbox(
            "📍 Select intervention point:",
            valid_range,
            key="its_intervention_pt",
            help="Choose the time point when the intervention occurred.",
        )

        idx_pos = list(series_preview.index).index(intervention_point)

        st.caption(
            f"Pre-intervention: **{idx_pos}** periods | "
            f"Post-intervention: **{len(series_preview) - idx_pos}** periods"
        )

    # ========================================================
    # Persist analysis after clicking Run Analysis
    # This fixes the ARIMA issue where clicking Fit ARIMA reruns the page.
    # ========================================================
    current_config = {
        "dataset_choice": dataset_choice,
        "time_col": time_col,
        "value_col": value_col,
        "freq_label": freq_label,
        "freq": int(freq) if isinstance(freq, (int, np.integer)) else freq,
        "analysis_type": analysis_type,
        "intervention_point": str(intervention_point),
        "control_col": control_col,
    }

    if "ts_last_config" not in st.session_state:
        st.session_state["ts_last_config"] = current_config

    if "ts_analysis_started" not in st.session_state:
        st.session_state["ts_analysis_started"] = False

    if st.session_state["ts_last_config"] != current_config:
        st.session_state["ts_analysis_started"] = False
        st.session_state["arima_results"] = None
        st.session_state["ts_last_config"] = current_config

    run_clicked = st.button(
        "▶ Run Analysis",
        use_container_width=True,
        key="ts_run"
    )

    if run_clicked:
        st.session_state["ts_analysis_started"] = True
        st.session_state["ts_last_config"] = current_config

    if st.session_state["ts_analysis_started"]:
        try:
            series = series_preview.copy()

            if len(series) < 8:
                st.error("Need at least 8 time points to run analysis.")
                return

            if "ITS" in analysis_type:
                if intervention_point is None:
                    st.error("Please select an intervention point.")
                    return

                control_series = None

                if control_col and control_col != "None":
                    ctrl_df = mdf[[time_col, control_col]].dropna().copy()

                    try:
                        ctrl_df[time_col] = pd.to_datetime(ctrl_df[time_col])
                    except Exception:
                        pass

                    try:
                        ctrl_df = ctrl_df.sort_values(time_col).set_index(time_col)
                    except Exception:
                        ctrl_df = ctrl_df.set_index(time_col)

                    control_series = ctrl_df[control_col]

                run_its(
                    series=series,
                    intervention_point=intervention_point,
                    plot_template=plot_template,
                    control_series=control_series
                )

            else:
                run_time_series(
                    series=series,
                    freq=freq,
                    plot_template=plot_template
                )

        except Exception as e:
            st.error(f"Time series analysis error: {e}")

            with st.expander("Error details"):
                import traceback
                st.code(traceback.format_exc())
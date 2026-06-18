# ============================================================
# modules/models/time_series.py
# Time Series Analysis + Interrupted Time Series (ITS)
# Full diagnostics + fix suggestions on the same page
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# statsmodels
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.outliers_influence import OLSInfluence
import statsmodels.api as sm
import pmdarima as pm


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

            components = [
                ("Observed", series.dropna().values, "#2563EB"),
                ("Trend", decomp.trend.values, "#16A34A"),
                ("Seasonal", decomp.seasonal.values, "#F59E0B"),
                ("Residual", decomp.resid.values, "#DC2626"),
            ]

            decomp_index = series.dropna().index

            fig_decomp = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.04,
                subplot_titles=[name for name, _, _ in components],
            )

            for i, (name, vals, color) in enumerate(components):
                fig_decomp.add_trace(
                    go.Scatter(
                        x=decomp_index,
                        y=vals,
                        mode="lines",
                        name=name,
                        line=dict(color=color, width=1.5),
                        showlegend=False,
                    ),
                    row=i + 1, col=1,
                )

            fig_decomp.update_layout(
                title=f"Decomposition ({decomp_model})",
                template=plot_template,
                height=700,
                margin=dict(t=80),
            )
            fig_decomp.update_xaxes(title_text="Time", row=4, col=1)

            st.plotly_chart(fig_decomp, use_container_width=True)

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

def render_ols_diagnostic_plots(result, plot_template):
    """
    Renders the four classic OLS regression diagnostic plots
    (matching R's plot(lm_model)): Residuals vs Fitted, Q-Q Residuals,
    Scale-Location, and Residuals vs Leverage (with Cook's distance).
    Only applicable to OLS-based models like the Segmented Regression
    ITS — not meaningful for ARIMA, which has its own residual checks.
    """
    st.markdown("### 📐 Regression Diagnostic Plots")
    st.caption(
        "Four standard checks for linear regression models — matching "
        "R's plot(lm_model). These are specific to OLS-based models "
        "(Segmented Regression) and are not shown for ARIMA, which uses "
        "different residual diagnostics (Ljung-Box, Durbin-Watson above)."
    )

    try:
        influence = OLSInfluence(result)
        fitted = np.asarray(result.fittedvalues)
        resid = np.asarray(result.resid)
        std_resid = np.asarray(influence.resid_studentized_internal)
        leverage = np.asarray(influence.hat_matrix_diag)
        cooks_d = np.asarray(influence.cooks_distance[0])
        n = len(resid)
    except Exception as e:
        st.info(f"Diagnostic plots could not be computed: {e}")
        return

    cooks_threshold = 4 / n
    high_cooks_idx = np.where(cooks_d > cooks_threshold)[0]

    col_a, col_b = st.columns(2)

    # ── 1. Residuals vs Fitted ──────────────────────────────────
    with col_a:
        st.markdown("**Residuals vs Fitted**")
        st.caption("Checks linearity. The smoothed line should stay flat near zero.")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=fitted, y=resid, mode="markers",
            marker=dict(size=5, color="#2563EB", opacity=0.6),
            name="Residuals",
        ))
        try:
            sort_idx = np.argsort(fitted)
            frac = max(0.3, min(0.9, 20 / n))
            lowess = sm.nonparametric.lowess(resid, fitted, frac=frac)
            fig1.add_trace(go.Scatter(
                x=lowess[:, 0], y=lowess[:, 1], mode="lines",
                line=dict(color="#DC2626", width=2), name="Trend",
            ))
        except Exception:
            pass
        fig1.add_hline(y=0, line_dash="dash", line_color="grey")
        fig1.update_layout(
            xaxis_title="Fitted values", yaxis_title="Residuals",
            template=plot_template, height=350,
        )
        st.plotly_chart(fig1, use_container_width=True)

    # ── 2. Q-Q Residuals ─────────────────────────────────────────
    with col_b:
        st.markdown("**Q-Q Residuals**")
        st.caption("Checks normality. Points should follow the dashed line closely.")
        osm, osr = stats.probplot(std_resid)[0]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(osm), y=list(osr), mode="markers",
            marker=dict(size=5, color="#2563EB", opacity=0.6),
            name="Residuals",
        ))
        fig2.add_trace(go.Scatter(
            x=[min(osm), max(osm)], y=[min(osm), max(osm)],
            mode="lines", line=dict(color="#DC2626", dash="dash"),
            name="Normal line",
        ))
        fig2.update_layout(
            xaxis_title="Theoretical Quantiles", yaxis_title="Standardized Residuals",
            template=plot_template, height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    # ── 3. Scale-Location ────────────────────────────────────────
    with col_c:
        st.markdown("**Scale-Location**")
        st.caption("Checks equal variance (homoscedasticity). Flat line = stable variance.")
        sqrt_std_resid = np.sqrt(np.abs(std_resid))
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=fitted, y=sqrt_std_resid, mode="markers",
            marker=dict(size=5, color="#2563EB", opacity=0.6),
            name="√|Standardized residuals|",
        ))
        try:
            frac = max(0.3, min(0.9, 20 / n))
            lowess3 = sm.nonparametric.lowess(sqrt_std_resid, fitted, frac=frac)
            fig3.add_trace(go.Scatter(
                x=lowess3[:, 0], y=lowess3[:, 1], mode="lines",
                line=dict(color="#DC2626", width=2), name="Trend",
            ))
        except Exception:
            pass
        fig3.update_layout(
            xaxis_title="Fitted values", yaxis_title="√|Standardized residuals|",
            template=plot_template, height=350,
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── 4. Residuals vs Leverage (Cook's Distance) ────────────────
    with col_d:
        st.markdown("**Residuals vs Leverage**")
        st.caption("Flags influential points. Watch for points beyond Cook's distance.")
        fig4 = go.Figure()
        point_colors = ["#DC2626" if cd > cooks_threshold else "#2563EB" for cd in cooks_d]
        fig4.add_trace(go.Scatter(
            x=leverage, y=std_resid, mode="markers",
            marker=dict(size=5, color=point_colors, opacity=0.7),
            name="Standardized residuals",
        ))
        fig4.add_hline(y=0, line_dash="dash", line_color="grey")
        fig4.update_layout(
            xaxis_title="Leverage", yaxis_title="Standardized Residuals",
            template=plot_template, height=350,
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Influential points summary ────────────────────────────────
    if len(high_cooks_idx) > 0:
        st.warning(
            f"🟡 {len(high_cooks_idx)} observation(s) exceed the Cook's distance "
            f"threshold (4/n = {cooks_threshold:.4f}), shown in red on the Leverage "
            f"plot. These points have an outsized influence on the model — "
            f"review them for data entry errors or genuine extreme events."
        )
        influence_tbl = pd.DataFrame({
            "Observation index": high_cooks_idx,
            "Cook's Distance": np.round(cooks_d[high_cooks_idx], 5),
            "Leverage": np.round(leverage[high_cooks_idx], 4),
            "Std. Residual": np.round(std_resid[high_cooks_idx], 3),
        }).sort_values("Cook's Distance", ascending=False)
        st.dataframe(influence_tbl, use_container_width=True)
    else:
        st.success("✅ No observations exceed the Cook's distance threshold.")


def run_its(series, intervention_point, plot_template, control_series=None, lag_k=0,
            adjust_seasonality=False, seasonal_freq=None):
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

    lag_k = int(lag_k) if lag_k else 0
    effect_start_idx = intervention_idx + lag_k

    if lag_k > 0:
        st.info(
            f"Effect onset delayed by **{lag_k}** period(s) after the intervention date. "
            f"The model treats periods between the intervention and the effect onset "
            f"as if the intervention had not yet taken effect."
        )

    intervention_flag = (time_var >= effect_start_idx + 1).astype(int)

    time_after = np.where(
        time_var >= effect_start_idx + 1,
        time_var - effect_start_idx,
        0,
    )

    pre_n = intervention_idx
    post_n = n - intervention_idx
    effective_post_n = n - effect_start_idx

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pre-intervention periods", pre_n)
    c2.metric("Post-intervention periods", post_n)
    c3.metric("Periods with active effect", effective_post_n,
              help="Post-intervention periods minus the lag window.")
    c4.metric("Total observations", n)

    if pre_n < 8:
        issues.append({
            "level": "error",
            "msg": f"Only {pre_n} pre-intervention periods — very low statistical power.",
            "fix": "At least 8–12 pre-intervention periods are recommended for reliable estimates.",
        })

    if effective_post_n < 4:
        issues.append({
            "level": "warning",
            "msg": f"Only {effective_post_n} period(s) with an active effect after the lag window — estimates may be unstable.",
            "fix": "Extend follow-up period, or reduce the lag, for more reliable estimates.",
        })

    # ── Seasonal dummy variables (matches R's seasonaldummy()) ──
    seasonal_cols = []
    n_seasonal_dummies = 0

    if adjust_seasonality and seasonal_freq and seasonal_freq >= 2 and n >= seasonal_freq * 2:
        seasonal_period = np.arange(n) % int(seasonal_freq)
        dummy_df_season = pd.get_dummies(
            seasonal_period, prefix="season", drop_first=True
        ).astype(float)
        seasonal_cols = list(dummy_df_season.columns)
        n_seasonal_dummies = len(seasonal_cols)
        st.info(
            f"Adjusting for seasonality with {n_seasonal_dummies} seasonal dummy variable(s) "
            f"(period = {int(seasonal_freq)}). This separates the seasonal pattern from the "
            f"intervention's true effect on level and slope."
        )
    elif adjust_seasonality:
        st.warning(
            "Not enough observations to build seasonal dummies for this period "
            "(need at least 2 full cycles). Seasonality adjustment was skipped."
        )

    base_cols = ["Time", "Intervention", "Time_After"]
    X_base = np.column_stack([time_var, intervention_flag, time_after])

    if seasonal_cols:
        X = sm.add_constant(
            np.column_stack([X_base, dummy_df_season.values])
        )
        all_col_names = ["const"] + base_cols + seasonal_cols
    else:
        X = sm.add_constant(X_base)
        all_col_names = ["const"] + base_cols

    X_df = pd.DataFrame(
        X,
        columns=all_col_names,
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
        + (" | Seasonal terms control for periodic patterns." if seasonal_cols else "")
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

    # Build readable parameter names — keep seasonal terms in a collapsible
    # section instead of cluttering the main table, since the course
    # treats them as nuisance/control variables (not of primary interest).
    main_parameter_names = [
        "Intercept (β₀)",
        "Pre-trend (β₁)",
        "Level change (β₂)",
        "Slope change (β₃)",
    ]
    n_main = len(main_parameter_names)

    coef_tbl = pd.DataFrame({
        "Parameter": main_parameter_names,
        "Coefficient": np.round(params[:n_main], 4),
        "Std Error": np.round(bse[:n_main], 4),
        "p-value": np.round(pvals[:n_main], 5),
        "CI Lower": np.round(ci_lo[:n_main], 4),
        "CI Upper": np.round(ci_hi[:n_main], 4),
        "Significant": ["✅" if p < 0.05 else "" for p in pvals[:n_main]],
    })

    st.dataframe(coef_tbl, use_container_width=True)

    st.download_button(
        "📥 Download ITS coefficients (CSV)",
        data=coef_tbl.to_csv(index=False).encode(),
        file_name="its_coefficients.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if seasonal_cols:
        with st.expander(f"Seasonal control coefficients ({n_seasonal_dummies} terms)"):
            st.caption(
                "These terms absorb the periodic seasonal pattern so it is not "
                "mistaken for the intervention's effect. They are not the focus "
                "of the analysis (similar to nuisance parameters)."
            )
            seasonal_tbl = pd.DataFrame({
                "Seasonal term": seasonal_cols,
                "Coefficient": np.round(params[n_main:n_main + n_seasonal_dummies], 4),
                "p-value": np.round(pvals[n_main:n_main + n_seasonal_dummies], 5),
            })
            st.dataframe(seasonal_tbl, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R²", round(result.rsquared, 4))
    m2.metric("Adj. R²", round(result.rsquared_adj, 4))
    m3.metric("AIC", round(float(result.aic), 2))
    m4.metric("BIC", round(float(result.bic), 2))

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

    if seasonal_cols:
        # Counterfactual keeps the TRUE seasonal pattern, only zeroes out
        # the intervention and time-after-intervention terms — matching
        # the course's predict(newdata) approach where `month=month` is kept.
        X_cf = np.column_stack([
            np.ones(n),
            time_var,
            np.zeros(n),
            np.zeros(n),
            dummy_df_season.values,
        ])
    else:
        X_cf = np.column_stack([
            np.ones(n),
            time_var,
            np.zeros(n),
            np.zeros(n),
        ])

    counterfactual = X_cf @ params[:X_cf.shape[1]]

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
    st.caption("Durbin-Watson checks autocorrelation at lag 1 only.")

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

    # ── Ljung-Box Test (checks multiple lags at once) ──────────
    st.markdown("**Ljung-Box Test** (checks autocorrelation across multiple lags)")
    st.caption(
        "Unlike Durbin-Watson, this can catch autocorrelation at seasonal lags "
        "(e.g. lag 12 for monthly data) that lag-1 tests might miss."
    )

    try:
        n_resid = len(result.resid)
        max_lb_lag = min(12, max(1, n_resid // 4))
        lb_lags = sorted(set([1, min(4, max_lb_lag), max_lb_lag]))
        lb_lags = [l for l in lb_lags if l >= 1 and l < n_resid]

        if lb_lags:
            lb_result = acorr_ljungbox(result.resid, lags=lb_lags, return_df=True)
            lb_display = lb_result.reset_index().rename(
                columns={"index": "Lag", "lb_stat": "Statistic", "lb_pvalue": "p-value"}
            )
            lb_display["Statistic"] = lb_display["Statistic"].round(4)
            lb_display["p-value"] = lb_display["p-value"].round(5)
            lb_display["Autocorrelated?"] = lb_display["p-value"].apply(
                lambda p: "⚠️ Yes" if p < 0.05 else "No"
            )
            st.dataframe(lb_display, use_container_width=True)

            sig_lags = lb_display[lb_display["p-value"] < 0.05]["Lag"].tolist()
            if sig_lags:
                issues.append({
                    "level": "warning",
                    "msg": f"Ljung-Box test detects residual autocorrelation at lag(s): {sig_lags}.",
                    "fix": (
                        "Consider adding seasonal terms, a different lag specification, "
                        "or modeling the residuals with ARIMA errors."
                    ),
                })
        else:
            st.info("Not enough residuals to run the Ljung-Box test at multiple lags.")

    except Exception as e:
        st.info(f"Ljung-Box test could not be performed: {e}")

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
            # Use SSR from the BASE model (time + intervention + time_after only)
            # for a fair structural-break comparison, even if seasonal dummies
            # were added to the main ITS model — otherwise ssr_full would be
            # artificially small and could even make the F-statistic negative.
            X_base_for_chow = sm.add_constant(X_base)
            ssr_full = sm.OLS(series.values, X_base_for_chow).fit().ssr

            # Chow test compares structural break using the BASE specification
            # (time, intervention, time_after) regardless of whether seasonal
            # terms were added — this keeps k consistent with the pre/post
            # sub-models and avoids artificially weakening the test when many
            # seasonal dummies are present.
            k = len(base_cols) + 1  # +1 for intercept
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

    render_ols_diagnostic_plots(result, plot_template)

    st.markdown("### 🩺 Diagnostic Summary")
    _diagnostic_summary(issues)

    with st.expander("Full ITS Model Summary"):
        st.text(result.summary().as_text())

    # ── Return key values for optional ARIMA comparison ────────
    return {
        "segmented_level": float(params[2]),
        "segmented_level_p": float(pvals[2]),
        "segmented_level_ci": (float(ci_lo[2]), float(ci_hi[2])),
        "segmented_slope": float(params[3]),
        "segmented_slope_p": float(pvals[3]),
        "segmented_slope_ci": (float(ci_lo[3]), float(ci_hi[3])),
        "segmented_aic": float(result.aic),
        "segmented_bic": float(result.bic),
        "intervention_flag": intervention_flag,
        "time_after": time_after,
    }


# ============================================================
# Compare Segmented Regression with ARIMA on the same intervention
# ============================================================

def fit_arima_auto(values, exog, seasonal_freq=None,
                   max_p=5, max_q=5, max_order=10):
    """
    Fits the best ARIMA model automatically using pmdarima.auto_arima,
    matching R's auto.arima(ts_data, xreg=..., seasonal=TRUE) approach.
    Returns a dict with the fitted model, order, AIC/BIC, and the
    intervention/time_after coefficients (x1, x2).
    """
    seasonal = bool(seasonal_freq and seasonal_freq >= 2)
    m = int(seasonal_freq) if seasonal else 1

    model = pm.auto_arima(
        values, X=exog,
        seasonal=seasonal, m=m,
        suppress_warnings=True, error_action="ignore",
        stepwise=True, max_p=max_p, max_q=max_q, max_order=max_order,
    )

    sm_result = model.arima_res_
    param_names = list(sm_result.param_names)

    level_idx = param_names.index("x1") if "x1" in param_names else None
    slope_idx = param_names.index("x2") if "x2" in param_names else None

    if level_idx is None or slope_idx is None:
        return None

    try:
        ci = sm_result.conf_int()
        level_ci = (float(ci[level_idx][0]), float(ci[level_idx][1]))
        slope_ci = (float(ci[slope_idx][0]), float(ci[slope_idx][1]))
    except Exception:
        level_ci = (np.nan, np.nan)
        slope_ci = (np.nan, np.nan)

    order_str = (
        f"ARIMA{model.order}" + (f"{model.seasonal_order}" if seasonal else "")
    )

    return {
        "model": model,
        "sm_result": sm_result,
        "order": model.order,
        "seasonal_order": model.seasonal_order if seasonal else None,
        "order_str": order_str,
        "aic": float(model.aic()),
        "bic": float(sm_result.bic),
        "level": float(sm_result.params[level_idx]),
        "level_p": float(sm_result.pvalues[level_idx]),
        "level_ci": level_ci,
        "slope": float(sm_result.params[slope_idx]),
        "slope_p": float(sm_result.pvalues[slope_idx]),
        "slope_ci": slope_ci,
    }


def render_arima_full_its(series, intervention_point, plot_template,
                          lag_k=0, adjust_seasonality=False, seasonal_freq=None):
    """
    Full ARIMA-based ITS analysis using auto_arima (matches R's auto.arima
    with xreg). Displays the same level of detail as the segmented
    regression: coefficients, CI, AIC/BIC, residual diagnostics.
    """
    st.markdown("## 📊 ARIMA-based ITS (auto-search)")
    st.caption(
        "Automatically searches for the best-fitting ARIMA(p,d,q)(P,D,Q) "
        "specification — matching R's auto.arima(xreg=...) — with the "
        "intervention and time-after vectors as exogenous regressors."
    )

    series = series.dropna()
    n = len(series)
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
            return None

    lag_k = int(lag_k) if lag_k else 0
    effect_start_idx = intervention_idx + lag_k

    time_var = np.arange(1, n + 1)
    intervention_flag = (time_var >= effect_start_idx + 1).astype(int)
    time_after = np.where(time_var >= effect_start_idx + 1, time_var - effect_start_idx, 0)
    exog = np.column_stack([intervention_flag, time_after])

    seasonal_freq_use = seasonal_freq if adjust_seasonality else None

    try:
        with st.spinner("Searching for the best ARIMA specification (auto_arima)..."):
            arima_results = fit_arima_auto(series.values, exog, seasonal_freq=seasonal_freq_use)
    except Exception as e:
        st.error(f"ARIMA auto-search failed: {e}")
        with st.expander("Error details"):
            import traceback
            st.code(traceback.format_exc())
        return None

    if arima_results is None:
        st.error("Could not extract intervention coefficients from the ARIMA model.")
        return None

    st.success(f"Selected model: **{arima_results['order_str']}**")

    m1, m2, m3 = st.columns(3)
    m1.metric("AIC", round(arima_results["aic"], 2))
    m2.metric("BIC", round(arima_results["bic"], 2))
    m3.metric("N", n)

    st.markdown("### ARIMA Coefficients — Intervention Effect")
    coef_tbl = pd.DataFrame({
        "Parameter": ["Level change (Intervention)", "Slope change (Time After)"],
        "Coefficient": [round(arima_results["level"], 4), round(arima_results["slope"], 6)],
        "p-value": [round(arima_results["level_p"], 5), round(arima_results["slope_p"], 5)],
        "CI Lower": [round(arima_results["level_ci"][0], 4), round(arima_results["slope_ci"][0], 6)],
        "CI Upper": [round(arima_results["level_ci"][1], 4), round(arima_results["slope_ci"][1], 6)],
        "Significant": [
            "✅" if arima_results["level_p"] < 0.05 else "",
            "✅" if arima_results["slope_p"] < 0.05 else "",
        ],
    })
    st.dataframe(coef_tbl, use_container_width=True)

    st.download_button(
        "📥 Download ARIMA coefficients (CSV)",
        data=coef_tbl.to_csv(index=False).encode(),
        file_name="its_arima_coefficients.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Fitted vs Observed + Counterfactual ─────────────────────
    st.markdown("### ARIMA Fit and Counterfactual")
    try:
        fitted_vals = arima_results["sm_result"].fittedvalues

        exog_cf = np.column_stack([np.zeros(n), np.zeros(n)])
        # Re-predict with zeroed intervention using the same fitted SARIMAX params
        model_cf = arima_results["model"]
        counterfactual = model_cf.predict_in_sample(X=exog_cf)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, mode="lines",
            name="Observed", line=dict(color="#2563EB", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=series.index, y=fitted_vals, mode="lines",
            name="ARIMA Fitted", line=dict(color="#16A34A", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=series.index, y=counterfactual, mode="lines",
            name="Counterfactual", line=dict(color="#DC2626", width=2, dash="dash"),
        ))
        fig.add_vline(
            x=series.index[intervention_idx] if hasattr(series.index, "__getitem__") else intervention_idx,
            line_dash="dash", line_color="#F59E0B",
        )
        fig.update_layout(
            title="ARIMA: Observed vs Fitted vs Counterfactual",
            xaxis_title="Time", yaxis_title="Value",
            template=plot_template,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Counterfactual plot could not be generated: {e}")

    # ── Residual diagnostics ─────────────────────────────────────
    st.markdown("### 🔬 Residual Diagnostics")
    try:
        resid = arima_results["sm_result"].resid
        dw = durbin_watson(resid)
        st.metric("Durbin-Watson", round(dw, 4), help="Ideal ≈ 2.0")

        lb_lags = sorted(set([1, min(4, max(1, n // 8)), min(12, max(1, n // 4))]))
        lb_lags = [l for l in lb_lags if l < n]
        if lb_lags:
            lb_result = acorr_ljungbox(resid, lags=lb_lags, return_df=True)
            lb_display = lb_result.reset_index().rename(
                columns={"index": "Lag", "lb_stat": "Statistic", "lb_pvalue": "p-value"}
            )
            lb_display["Autocorrelated?"] = lb_display["p-value"].apply(
                lambda p: "⚠️ Yes" if p < 0.05 else "No"
            )
            st.dataframe(lb_display, use_container_width=True)
    except Exception as e:
        st.info(f"Residual diagnostics could not be computed: {e}")

    with st.expander("Full ARIMA Model Summary"):
        st.text(arima_results["sm_result"].summary().as_text())

    return {
        "arima_level": arima_results["level"],
        "arima_level_p": arima_results["level_p"],
        "arima_level_ci": arima_results["level_ci"],
        "arima_slope": arima_results["slope"],
        "arima_slope_p": arima_results["slope_p"],
        "arima_slope_ci": arima_results["slope_ci"],
        "arima_aic": arima_results["aic"],
        "arima_bic": arima_results["bic"],
        "arima_order_str": arima_results["order_str"],
    }


def render_method_comparison_its(segmented_results, arima_results, plot_template):
    """
    Always-shown comparison table between Segmented Regression and
    auto-search ARIMA, regardless of which method the user selected
    as the primary method.
    """
    if not segmented_results or not arima_results:
        st.info("Comparison unavailable — one of the two models could not be fitted.")
        return

    st.markdown("## ⚖️ Method Comparison: Segmented Regression vs ARIMA")
    st.caption(
        "Shown automatically regardless of which method you selected as primary — "
        "useful as a robustness check, following the course's approach of "
        "comparing both methods before deciding which one to report."
    )

    compare_df = pd.DataFrame({
        "Method": ["Segmented Regression", arima_results["arima_order_str"]],
        "Level change": [
            round(segmented_results["segmented_level"], 4),
            round(arima_results["arima_level"], 4),
        ],
        "Level p-value": [
            round(segmented_results["segmented_level_p"], 5),
            round(arima_results["arima_level_p"], 5),
        ],
        "Slope change": [
            round(segmented_results["segmented_slope"], 6),
            round(arima_results["arima_slope"], 6),
        ],
        "Slope p-value": [
            round(segmented_results["segmented_slope_p"], 5),
            round(arima_results["arima_slope_p"], 5),
        ],
        "AIC": [
            round(segmented_results["segmented_aic"], 2),
            round(arima_results["arima_aic"], 2),
        ],
    })

    st.dataframe(compare_df, use_container_width=True)

    st.download_button(
        "📥 Download method comparison (CSV)",
        data=compare_df.to_csv(index=False).encode(),
        file_name="its_method_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )

    seg_sig = segmented_results["segmented_level_p"] < 0.05
    arima_sig = arima_results["arima_level_p"] < 0.05
    better_aic = "Segmented Regression" if segmented_results["segmented_aic"] < arima_results["arima_aic"] else arima_results["arima_order_str"]

    st.metric("Better fit by AIC (lower is better)", better_aic)

    if seg_sig and arima_sig:
        st.success(
            "✅ Both methods agree: the level change is statistically significant. "
            "This agreement strengthens confidence in the result."
        )
    elif not seg_sig and not arima_sig:
        st.info(
            "Both methods agree: the level change is not statistically significant."
        )
    else:
        st.warning(
            "🟡 The two methods disagree on statistical significance for the level "
            "change. As the course material notes, segmented regression is "
            "preferable for relatively simple series, while ARIMA is more "
            "appropriate when there is unusual trend or significant "
            "autocorrelation — review the residual diagnostics for each "
            "method to decide which better fits this series."
        )


# ============================================================
# Robustness check: Placebo test on multiple fake intervention points
# ============================================================

def run_lag_comparison_its(series, intervention_point, plot_template,
                           max_lag=6, adjust_seasonality=False, seasonal_freq=None):
    """
    Fits the ITS model at several lag values (0, 1, 2, ..., max_lag) and
    compares AIC, level change, and slope change across them — matching
    the course/assignment approach of comparing an immediate-effect model
    with a lagged-effect model using AIC to decide which fits better.
    """
    st.markdown("### ⏱ Compare Effect Lag Values")
    st.caption(
        "Fits the ITS model at several lag values and compares AIC, level "
        "change, and slope change side-by-side — useful for deciding whether "
        "an immediate or a delayed effect specification fits the data better. "
        "Lower AIC indicates a better-fitting model, but differences should be "
        "interpreted alongside statistical significance and residual diagnostics."
    )

    series = series.dropna()
    n = len(series)
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
            st.info("Could not locate the intervention point for lag comparison.")
            return

    time_var = np.arange(1, n + 1)

    seasonal_dummy_vals = None
    if adjust_seasonality and seasonal_freq and seasonal_freq >= 2 and n >= seasonal_freq * 2:
        seasonal_period = np.arange(n) % int(seasonal_freq)
        seasonal_dummy_vals = pd.get_dummies(
            seasonal_period, prefix="season", drop_first=True
        ).astype(float).values

    rows = []
    for lag_k in range(0, int(max_lag) + 1):
        eff_start = intervention_idx + lag_k
        if eff_start >= n - 2:
            continue

        flag = (time_var >= eff_start + 1).astype(int)
        t_after = np.where(time_var >= eff_start + 1, time_var - eff_start, 0)

        if seasonal_dummy_vals is not None:
            X_lag = sm.add_constant(
                np.column_stack([time_var, flag, t_after, seasonal_dummy_vals])
            )
        else:
            X_lag = sm.add_constant(np.column_stack([time_var, flag, t_after]))

        try:
            maxlags_hac = max(1, min(4, intervention_idx // 2))
            result_lag = sm.OLS(series.values, X_lag).fit(
                cov_type="HAC", cov_kwds={"maxlags": maxlags_hac}
            )
            rows.append({
                "Lag (periods)": lag_k,
                "Level change": round(float(result_lag.params[2]), 4),
                "Level p-value": round(float(result_lag.pvalues[2]), 5),
                "Slope change": round(float(result_lag.params[3]), 6),
                "Slope p-value": round(float(result_lag.pvalues[3]), 5),
                "AIC": round(float(result_lag.aic), 2),
                "BIC": round(float(result_lag.bic), 2),
            })
        except Exception:
            continue

    if not rows:
        st.info("Could not fit any lag value — check that there is enough post-intervention data.")
        return

    lag_df = pd.DataFrame(rows)
    st.dataframe(lag_df, use_container_width=True)

    st.download_button(
        "📥 Download lag comparison table (CSV)",
        data=lag_df.to_csv(index=False).encode(),
        file_name="its_lag_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lag_df["Lag (periods)"], y=lag_df["AIC"],
        mode="lines+markers", name="AIC",
        line=dict(color="#2563EB", width=2),
        marker=dict(size=10),
    ))
    best_row = lag_df.loc[lag_df["AIC"].idxmin()]
    fig.add_trace(go.Scatter(
        x=[best_row["Lag (periods)"]], y=[best_row["AIC"]],
        mode="markers", name="Lowest AIC",
        marker=dict(size=16, color="#16A34A", symbol="star"),
    ))
    fig.update_layout(
        title="AIC by Effect Lag",
        xaxis_title="Lag (periods after intervention)",
        yaxis_title="AIC",
        template=plot_template,
    )
    st.plotly_chart(fig, use_container_width=True)

    best_lag = int(best_row["Lag (periods)"])
    aic_range = lag_df["AIC"].max() - lag_df["AIC"].min()

    if aic_range < 2:
        st.info(
            f"AIC values are very close across all tested lags (range = {aic_range:.2f}). "
            f"No lag specification stands out as clearly better — consider using domain "
            f"knowledge (e.g. how quickly the intervention could realistically take effect) "
            f"to choose the most defensible lag, rather than relying on AIC alone."
        )
    else:
        st.success(
            f"Lowest AIC is at lag = {best_lag} periods. This suggests the effect may "
            f"realistically begin {best_lag} period(s) after the intervention date, "
            f"though this should be interpreted alongside the significance and "
            f"residual diagnostics of that specific model."
        )


def run_placebo_test_its(series, intervention_point, lag_k, plot_template, n_placebo=6, safety_gap=4):
    """
    Robustness check: re-run the ITS model at several FAKE intervention points
    located BEFORE the real intervention. If the model is sound, most of these
    placebo points should show a non-significant level change, while the real
    intervention point should stand out as significant.
    """
    st.markdown("### 🧪 Robustness Check — Placebo Test (Multiple Fake Intervention Points)")
    st.caption(
        "Re-runs the same ITS model at several fake intervention points located "
        "before the real one. If most fake points are non-significant while the "
        "real point is significant, this supports the credibility of the result. "
        "If many fake points are also significant, the series may simply have a "
        "strong underlying trend that can produce false positives anywhere."
    )

    series = series.dropna()
    n = len(series)
    index_list = list(series.index)

    try:
        real_idx = index_list.index(intervention_point)
    except ValueError:
        try:
            real_idx = min(
                range(len(index_list)),
                key=lambda i: abs(pd.Timestamp(index_list[i]) - pd.Timestamp(intervention_point))
            )
        except Exception:
            st.info("Could not locate the intervention point for the placebo test.")
            return

    min_placebo_idx = 6
    max_placebo_idx = real_idx - safety_gap

    if max_placebo_idx <= min_placebo_idx:
        st.info(
            "Not enough pre-intervention periods to run a placebo test "
            "(need extra room before the real intervention point)."
        )
        return

    step = max(1, (max_placebo_idx - min_placebo_idx) // n_placebo)
    placebo_indices = list(range(min_placebo_idx, max_placebo_idx, step))[:n_placebo]

    if len(placebo_indices) < 3:
        st.info("Not enough usable placebo points given the available pre-intervention data.")
        return

    time_var = np.arange(1, n + 1)
    results_rows = []

    def _fit_at(idx):
        eff_start = idx + lag_k
        flag = (time_var >= eff_start + 1).astype(int)
        t_after = np.where(time_var >= eff_start + 1, time_var - eff_start, 0)
        X_p = sm.add_constant(np.column_stack([time_var, flag, t_after]))
        r = sm.OLS(series.values, X_p).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        return float(r.params[2]), float(r.pvalues[2])

    for idx in placebo_indices:
        try:
            level, pval = _fit_at(idx)
            results_rows.append({
                "Point type": "Placebo (fake)",
                "Index position": idx,
                "Date / label": str(index_list[idx]),
                "Level change": round(level, 4),
                "p-value": round(pval, 5),
                "Significant?": "⚠️ Yes" if pval < 0.05 else "No",
            })
        except Exception:
            continue

    try:
        real_level, real_pval = _fit_at(real_idx)
        results_rows.append({
            "Point type": "REAL intervention",
            "Index position": real_idx,
            "Date / label": str(intervention_point),
            "Level change": round(real_level, 4),
            "p-value": round(real_pval, 5),
            "Significant?": "✅ Yes" if real_pval < 0.05 else "No",
        })
    except Exception as e:
        st.info(f"Could not fit the real intervention point for comparison: {e}")
        return

    results_df = pd.DataFrame(results_rows)
    st.dataframe(results_df, use_container_width=True)

    fig = go.Figure()
    placebo_rows = results_df[results_df["Point type"] == "Placebo (fake)"]
    real_row = results_df[results_df["Point type"] == "REAL intervention"]

    fig.add_trace(go.Scatter(
        x=placebo_rows["Index position"], y=placebo_rows["p-value"],
        mode="markers", name="Placebo points",
        marker=dict(size=10, color="#94A3B8"),
    ))
    fig.add_trace(go.Scatter(
        x=real_row["Index position"], y=real_row["p-value"],
        mode="markers", name="Real intervention",
        marker=dict(size=14, color="#DC2626", symbol="star"),
    ))
    fig.add_hline(y=0.05, line_dash="dash", line_color="orange",
                 annotation_text="p = 0.05")
    fig.update_layout(
        title="Placebo Test — p-value by Intervention Point",
        xaxis_title="Time index of (fake or real) intervention point",
        yaxis_title="p-value (level change)",
        template=plot_template,
    )
    st.plotly_chart(fig, use_container_width=True)

    n_placebo_sig = (placebo_rows["Significant?"] == "⚠️ Yes").sum()
    n_placebo_total = len(placebo_rows)
    pct_sig = round(n_placebo_sig / n_placebo_total * 100, 1) if n_placebo_total > 0 else 0

    real_is_sig = (real_row["Significant?"].iloc[0] == "✅ Yes") if not real_row.empty else False

    if real_is_sig and pct_sig <= 20:
        st.success(
            f"✅ The real intervention point is significant, and only {pct_sig}% of "
            f"placebo points are significant. This supports the credibility of the result."
        )
    elif real_is_sig and pct_sig > 50:
        st.warning(
            f"🟡 The real intervention point is significant, but {pct_sig}% of placebo "
            f"points are ALSO significant. This series may have a strong underlying trend "
            f"that produces significant-looking effects almost anywhere — interpret the "
            f"real result with caution."
        )
    elif not real_is_sig:
        st.info(
            "The real intervention point itself was not significant in this check "
            "(note: HAC standard errors here may differ slightly from the main model)."
        )
    else:
        st.info(f"{pct_sig}% of placebo points were significant. Review the table above for details.")


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
    lag_k = 0
    run_placebo = False
    adjust_seasonality = False
    primary_method = "Segmented Regression"
    run_lag_compare = False
    max_lag_compare = 6

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

        its3, its4 = st.columns(2)

        with its3:
            lag_k = st.number_input(
                "Effect lag (periods after intervention before the effect starts)",
                min_value=0,
                max_value=24,
                value=0,
                step=1,
                key="its_lag_k",
                help=(
                    "Default is 0 (effect starts immediately at the intervention date). "
                    "Increase this if you expect a delayed effect — e.g. enforcement of a "
                    "new law may take a few months to show measurable impact."
                ),
            )
            if lag_k > 0:
                st.caption(
                    f"The effect will be modeled as starting {lag_k} period(s) "
                    f"after the intervention date, not immediately."
                )

        with its4:
            run_placebo = st.checkbox(
                "Run placebo robustness check",
                value=False,
                key="its_run_placebo",
                help=(
                    "Re-runs the model at several fake intervention points before the "
                    "real one, to check whether the result could be a false positive "
                    "driven by the series' underlying trend."
                ),
            )

        st.markdown("---")
        st.markdown("**Primary estimation method**")
        st.caption(
            "Choose which method to view in full detail (coefficients, "
            "counterfactual plot, residual diagnostics). A comparison "
            "between both methods is always shown afterward, regardless "
            "of your choice — ARIMA uses automatic order search "
            "(matching R's auto.arima), not manually chosen parameters."
        )
        primary_method = st.radio(
            "Primary method",
            ["Segmented Regression", "ARIMA (auto-search)"],
            horizontal=True,
            key="its_primary_method",
            label_visibility="collapsed",
        )

        its5, its6 = st.columns(2)

        with its5:
            adjust_seasonality = st.checkbox(
                "Adjust for seasonality (seasonal dummies)",
                value=False,
                key="its_adjust_seasonality",
                help=(
                    "Adds seasonal dummy variables (e.g. month-of-year) to separate "
                    "the seasonal pattern from the intervention's true effect on "
                    "level and slope. Recommended for monthly/quarterly health data "
                    "with a known seasonal cycle. Applies to both methods."
                ),
            )
            if adjust_seasonality:
                st.caption(
                    f"Will use a seasonal period of {freq if freq else 'auto-detected'} "
                    f"(from the seasonality setting above)."
                )

        with its6:
            run_placebo_dummy_for_layout = None  # placeholder to keep column balanced

        its7, its8 = st.columns(2)

        with its7:
            run_lag_compare = st.checkbox(
                "Compare different lag values",
                value=False,
                key="its_run_lag_compare",
                help=(
                    "Fits the ITS model at lag = 0, 1, 2, ... and compares AIC "
                    "side-by-side, to help decide whether an immediate or "
                    "delayed effect specification fits the data better."
                ),
            )
            if run_lag_compare:
                max_lag_compare = st.slider(
                    "Maximum lag to test", 1, 12, 6, key="its_max_lag_compare",
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
- **Effect lag:** Some interventions take time to show an effect — set this above if relevant
- **Placebo test:** A robustness check using fake intervention points before the real one

**Controlled ITS** uses a parallel control series to remove common trends.
            """)

    ts_df = mdf[[time_col, value_col]].dropna().copy()

    if ts_df.empty:
        st.warning("No usable rows after removing missing values from the selected time/value columns.")
        return

    date_conversion_succeeded = False
    try:
        converted_dates = pd.to_datetime(ts_df[time_col], errors="raise")
        ts_df[time_col] = converted_dates
        date_conversion_succeeded = True
    except Exception:
        date_conversion_succeeded = False

    if date_conversion_succeeded:
        # Safe to sort chronologically using the parsed datetime values.
        ts_df = ts_df.sort_values(time_col).set_index(time_col)
    else:
        # Date parsing failed (e.g. mixed/ambiguous formats like "Jan-00"
        # and "5-Oct" in the same column). Sorting the raw strings would
        # scramble the chronological order (alphabetical instead of
        # time-based), which silently corrupts every downstream model.
        # The safest fallback is to preserve the file's original row
        # order, which is almost always already chronological for time
        # series data — exactly like keeping seq_len(nrow(df)) in R
        # instead of re-sorting a malformed date column.
        st.warning(
            "Could not parse '" + str(time_col) + "' as a date (mixed or "
            "ambiguous format detected). Using the original row order from "
            "the file as the time sequence instead of sorting — please "
            "confirm your file is already sorted chronologically."
        )
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
        "lag_k": int(lag_k),
        "run_placebo": run_placebo,
        "adjust_seasonality": adjust_seasonality,
        "primary_method": primary_method,
        "run_lag_compare": run_lag_compare,
        "max_lag_compare": int(max_lag_compare),
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

                    ctrl_date_ok = False
                    try:
                        ctrl_df[time_col] = pd.to_datetime(ctrl_df[time_col], errors="raise")
                        ctrl_date_ok = True
                    except Exception:
                        ctrl_date_ok = False

                    if ctrl_date_ok:
                        ctrl_df = ctrl_df.sort_values(time_col).set_index(time_col)
                    else:
                        # Same safeguard as the main series: don't sort
                        # unparseable date strings alphabetically.
                        ctrl_df = ctrl_df.set_index(time_col)

                    control_series = ctrl_df[control_col]

                seasonal_freq_for_its = freq if (adjust_seasonality and freq) else None

                segmented_results = None
                arima_results = None

                if primary_method == "Segmented Regression":
                    # ── Primary: Segmented Regression (full detail) ────
                    segmented_results = run_its(
                        series=series,
                        intervention_point=intervention_point,
                        plot_template=plot_template,
                        control_series=control_series,
                        lag_k=lag_k,
                        adjust_seasonality=adjust_seasonality,
                        seasonal_freq=seasonal_freq_for_its,
                    )

                    if run_placebo:
                        st.markdown("---")
                        run_placebo_test_its(
                            series=series,
                            intervention_point=intervention_point,
                            lag_k=lag_k,
                            plot_template=plot_template,
                        )

                    # ── Always fit ARIMA in the background for comparison ──
                    st.markdown("---")
                    try:
                        time_var_cmp = np.arange(1, len(series.dropna()) + 1)
                        idx_list_cmp = list(series.dropna().index)
                        try:
                            ivt_idx_cmp = idx_list_cmp.index(intervention_point)
                        except ValueError:
                            ivt_idx_cmp = min(
                                range(len(idx_list_cmp)),
                                key=lambda i: abs(pd.Timestamp(idx_list_cmp[i]) - pd.Timestamp(intervention_point))
                            )
                        eff_start_cmp = ivt_idx_cmp + int(lag_k)
                        flag_cmp = (time_var_cmp >= eff_start_cmp + 1).astype(int)
                        t_after_cmp = np.where(time_var_cmp >= eff_start_cmp + 1, time_var_cmp - eff_start_cmp, 0)
                        exog_cmp = np.column_stack([flag_cmp, t_after_cmp])
                        arima_fit_cmp = fit_arima_auto(
                            series.dropna().values, exog_cmp, seasonal_freq=seasonal_freq_for_its,
                        )
                        if arima_fit_cmp:
                            arima_results = {
                                "arima_level": arima_fit_cmp["level"],
                                "arima_level_p": arima_fit_cmp["level_p"],
                                "arima_level_ci": arima_fit_cmp["level_ci"],
                                "arima_slope": arima_fit_cmp["slope"],
                                "arima_slope_p": arima_fit_cmp["slope_p"],
                                "arima_slope_ci": arima_fit_cmp["slope_ci"],
                                "arima_aic": arima_fit_cmp["aic"],
                                "arima_bic": arima_fit_cmp["bic"],
                                "arima_order_str": arima_fit_cmp["order_str"],
                            }
                    except Exception as e:
                        st.info(f"Background ARIMA fit for comparison could not be completed: {e}")

                else:
                    # ── Primary: ARIMA (auto-search, full detail) ───────
                    arima_results = render_arima_full_its(
                        series=series,
                        intervention_point=intervention_point,
                        plot_template=plot_template,
                        lag_k=lag_k,
                        adjust_seasonality=adjust_seasonality,
                        seasonal_freq=seasonal_freq_for_its,
                    )

                    # ── Always fit Segmented Regression in the background ──
                    st.markdown("---")
                    try:
                        # Re-run run_its but suppress its own UI by capturing
                        # only the return value is not possible without UI,
                        # so we show it collapsed for transparency.
                        with st.expander("Segmented Regression details (background fit for comparison)"):
                            segmented_results = run_its(
                                series=series,
                                intervention_point=intervention_point,
                                plot_template=plot_template,
                                control_series=control_series,
                                lag_k=lag_k,
                                adjust_seasonality=adjust_seasonality,
                                seasonal_freq=seasonal_freq_for_its,
                            )
                    except Exception as e:
                        st.info(f"Background Segmented Regression fit could not be completed: {e}")

                    if run_placebo:
                        st.markdown("---")
                        run_placebo_test_its(
                            series=series,
                            intervention_point=intervention_point,
                            lag_k=lag_k,
                            plot_template=plot_template,
                        )

                # ── Always show the comparison, regardless of primary choice ──
                if segmented_results and arima_results:
                    st.markdown("---")
                    render_method_comparison_its(
                        segmented_results, arima_results, plot_template,
                    )

                if run_lag_compare:
                    st.markdown("---")
                    seasonal_freq_for_lag = freq if (adjust_seasonality and freq) else None
                    run_lag_comparison_its(
                        series=series,
                        intervention_point=intervention_point,
                        plot_template=plot_template,
                        max_lag=int(max_lag_compare),
                        adjust_seasonality=adjust_seasonality,
                        seasonal_freq=seasonal_freq_for_lag,
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
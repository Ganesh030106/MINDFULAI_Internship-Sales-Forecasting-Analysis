"""
pages/2_Forecast.py — Prophet-powered interactive sales forecast
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from prophet import Prophet
from utils import load_raw, sidebar_filters, inject_css, section, fmt_k

st.set_page_config(page_title="Sales Forecast", page_icon="🔮", layout="wide")
inject_css()

df_raw = load_raw()
df     = sidebar_filters(df_raw)

st.markdown("## 🔮 Sales Forecast")
st.caption("Facebook Prophet time-series model with confidence bands and seasonality decomposition.")

if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

# ── Forecast controls ───────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
horizon     = c1.selectbox("Forecast Horizon", [30, 60, 90], index=1,
                            format_func=lambda x: f"{x} days")
seasonality = c2.selectbox("Seasonality Mode", ["additive", "multiplicative"])
changepoint = c3.slider("Changepoint Sensitivity", 0.01, 0.5, 0.15, 0.01)

st.markdown("<hr>", unsafe_allow_html=True)

@st.cache_data(show_spinner="Training Prophet model…")
def run_prophet(_df, horizon, seasonality, changepoint):
    df_ts = (
        _df.groupby("Order Date")["Sales"]
        .sum().reset_index()
        .rename(columns={"Order Date": "ds", "Sales": "y"})
        .dropna()
    )
    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=True,
        daily_seasonality=False, seasonality_mode=seasonality,
        changepoint_prior_scale=changepoint,
    )
    model.fit(df_ts)
    future   = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    return df_ts, forecast

if len(df["Order Date"].unique()) < 30:
    st.warning("Not enough date points. Please widen the date range filter.")
    st.stop()

df_ts, forecast = run_prophet(df, horizon, seasonality, changepoint)

hist_end  = df_ts["ds"].max()
fc_future = forecast[forecast["ds"] > hist_end]

section(f"📈 {horizon}-Day Forecast with Confidence Bands")
fig = go.Figure()
# Confidence band
fig.add_trace(go.Scatter(
    x=pd.concat([forecast["ds"], forecast["ds"].iloc[::-1]]),
    y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"].iloc[::-1]]),
    fill="toself", fillcolor="rgba(108,99,255,0.12)",
    line=dict(color="rgba(255,255,255,0)"), name="Confidence Band", hoverinfo="skip",
))
# Historical
fig.add_trace(go.Scatter(
    x=df_ts["ds"], y=df_ts["y"], mode="lines", name="Historical",
    line=dict(color="#00C9A7", width=2),
))
# Model fit on history
fig.add_trace(go.Scatter(
    x=forecast[forecast["ds"] <= hist_end]["ds"],
    y=forecast[forecast["ds"] <= hist_end]["yhat"],
    mode="lines", name="Model Fit",
    line=dict(color="#6C63FF", width=1.5, dash="dot"),
))
# Future forecast
fig.add_trace(go.Scatter(
    x=fc_future["ds"], y=fc_future["yhat"],
    mode="lines", name=f"{horizon}d Forecast",
    line=dict(color="#FF6F61", width=2.5),
))
fig.add_vline(x=hist_end, line_dash="dash", line_color="rgba(255,255,255,0.3)",
              annotation_text="Forecast Start", annotation_position="top right")
fig.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
    margin=dict(l=0,r=0,t=10,b=0), height=420,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    yaxis_tickprefix="₹", yaxis_tickformat=",.0f",
)
st.plotly_chart(fig, use_container_width=True)

# ── Forecast summary ─────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
section("📋 Forecast Summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Total Forecasted ({horizon}d)", fmt_k(fc_future["yhat"].sum()))
m2.metric("Daily Average",                  fmt_k(fc_future["yhat"].mean()))
peak_row = fc_future.loc[fc_future["yhat"].idxmax()]
m3.metric("Peak Day",  peak_row["ds"].strftime("%d %b %Y"))
m4.metric("Peak Value", fmt_k(peak_row["yhat"]))

# ── Seasonality decomposition ─────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
section("🌀 Seasonality Decomposition")
tab_yr, tab_wk = st.tabs(["📆 Yearly Pattern", "📅 Weekly Pattern"])

with tab_yr:
    if "yearly" in forecast.columns:
        yr = forecast[["ds","yearly"]].drop_duplicates("ds").sort_values("ds")
        fig_yr = px.line(yr, x="ds", y="yearly",
                         labels={"ds":"Date","yearly":"Effect"},
                         color_discrete_sequence=["#FFD166"])
        fig_yr.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=0,r=0,t=0,b=0), height=240)
        st.plotly_chart(fig_yr, use_container_width=True)
    else:
        st.info("Yearly component not available.")

with tab_wk:
    if "weekly" in forecast.columns:
        wk = forecast[["ds","weekly"]].copy()
        wk["Weekday"] = wk["ds"].dt.day_name()
        wk_avg = (wk.groupby("Weekday")["weekly"].mean()
                  .reindex(["Monday","Tuesday","Wednesday","Thursday",
                             "Friday","Saturday","Sunday"])
                  .reset_index())
        fig_wk = px.bar(wk_avg, x="Weekday", y="weekly",
                        color="weekly", color_continuous_scale="Purples",
                        labels={"weekly":"Effect"})
        fig_wk.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
                              margin=dict(l=0,r=0,t=0,b=0), height=240)
        st.plotly_chart(fig_wk, use_container_width=True)
    else:
        st.info("Weekly component not available.")

# ── Download ─────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
dl_df = fc_future[["ds","yhat","yhat_lower","yhat_upper"]].rename(
    columns={"ds":"Date","yhat":"Forecast","yhat_lower":"Lower","yhat_upper":"Upper"})
st.download_button("⬇ Download Forecast CSV",
                   dl_df.to_csv(index=False).encode(),
                   f"forecast_{horizon}d.csv", "text/csv")

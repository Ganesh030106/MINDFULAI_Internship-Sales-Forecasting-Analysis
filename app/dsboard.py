"""
app/dsboard.py  — Main entrypoint
Redirects to the Overview page (Streamlit multi-page apps load this first).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from utils import load_raw, sidebar_filters, inject_css, section, fmt_k, fmt_inr, PALETTE
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="MINDFULAI — Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

df_raw  = load_raw()
df      = sidebar_filters(df_raw)

# ── Hero banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
            border-radius:16px;padding:32px 36px;margin-bottom:28px;
            border:1px solid rgba(108,99,255,0.3)">
  <h1 style="margin:0;font-size:2rem;font-weight:700;
             background:linear-gradient(90deg,#6C63FF,#4CC9F0);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent">
    📊 MINDFULAI Sales Intelligence
  </h1>
  <p style="margin:8px 0 0;color:#aaa;font-size:1rem">
    Superstore Sales · Forecasting · ML Insights · Data Explorer
  </p>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ──────────────────────────────────────────────────────────────────
total_sales  = df["Sales"].sum()
order_count  = df["Order ID"].nunique() if "Order ID" in df else len(df)
avg_sale     = df["Sales"].mean()
max_sale     = df["Sales"].max()
unique_cust  = df["Customer ID"].nunique() if "Customer ID" in df else 0

# MoM delta
latest_m = df["Order Date"].max()
cur  = df[df["Order Date"].dt.to_period("M") == latest_m.to_period("M")]["Sales"].sum()
prev = df[df["Order Date"].dt.to_period("M") == (latest_m - pd.DateOffset(months=1)).to_period("M")]["Sales"].sum()
delta_pct = ((cur - prev) / prev * 100) if prev else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("💰 Total Sales",    fmt_k(total_sales))
c2.metric("🛒 Orders",         f"{order_count:,}")
c3.metric("📦 Avg Sale",       fmt_k(avg_sale))
c4.metric("🏆 Max Sale",       fmt_k(max_sale))
c5.metric("👤 Customers",      f"{unique_cust:,}", delta=f"{delta_pct:+.1f}% MoM")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Monthly sales sparkline ───────────────────────────────────────────────────
section("📈 Monthly Sales Trend")
monthly = (
    df.groupby(df["Order Date"].dt.to_period("M"))["Sales"]
    .sum()
    .reset_index()
)
monthly["Order Date"] = monthly["Order Date"].dt.to_timestamp()

fig_spark = px.area(
    monthly, x="Order Date", y="Sales",
    labels={"Sales": "Sales (₹)", "Order Date": "Month"},
    color_discrete_sequence=["#6C63FF"],
)
fig_spark.update_traces(
    fill="tozeroy",
    fillcolor="rgba(108,99,255,0.15)",
    line=dict(width=2.5),
)
fig_spark.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0),
    hovermode="x unified", height=280,
)
st.plotly_chart(fig_spark, use_container_width=True)

# ── Two-column: Category & Region ───────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    section("📦 Sales by Category")
    cat_df = df.groupby("Category")["Sales"].sum().reset_index().sort_values("Sales")
    fig_cat = px.bar(
        cat_df, x="Sales", y="Category", orientation="h",
        color="Sales", color_continuous_scale="Purples",
        labels={"Sales": "Sales (₹)"},
    )
    fig_cat.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=0, b=0), height=260,
    )
    st.plotly_chart(fig_cat, use_container_width=True)

with col_b:
    section("🌐 Sales by Region")
    reg_df = df.groupby("Region")["Sales"].sum().reset_index()
    fig_reg = px.pie(
        reg_df, names="Region", values="Sales",
        color_discrete_sequence=PALETTE, hole=0.55,
    )
    fig_reg.update_traces(textinfo="percent+label")
    fig_reg.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), height=260,
        showlegend=False,
    )
    st.plotly_chart(fig_reg, use_container_width=True)

# ── Quarterly heatmap ─────────────────────────────────────────────────────────
section("🗓 Quarterly Sales Heatmap")
heat = df.groupby(["Year", "Quarter"])["Sales"].sum().reset_index()
fig_heat = px.density_heatmap(
    heat, x="Quarter", y="Year", z="Sales",
    color_continuous_scale="Viridis", text_auto=".2s",
    labels={"Sales": "Sales (₹)"},
)
fig_heat.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0), height=260,
)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Navigate hint ────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:rgba(108,99,255,0.08);border:1px solid rgba(108,99,255,0.2);
            border-radius:12px;padding:16px 20px;margin-top:12px;text-align:center">
  <span style="color:#aaa">Use the <b style="color:#6C63FF">sidebar pages</b> to explore
  Sales Analytics, Forecasting, ML Insights, and the Data Explorer →</span>
</div>
""", unsafe_allow_html=True)

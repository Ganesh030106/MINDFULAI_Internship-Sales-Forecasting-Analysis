"""
pages/4_Data_Explorer.py — Searchable, sortable data table with download
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.express as px
import pandas as pd
from utils import load_raw, sidebar_filters, inject_css, section, fmt_inr, PALETTE

st.set_page_config(page_title="Data Explorer", page_icon="🔍", layout="wide")
inject_css()

df_raw = load_raw()
df     = sidebar_filters(df_raw)

st.markdown("## 🔍 Data Explorer")
st.caption("Search, sort and download the raw filtered dataset. Inspect individual records and summary statistics.")

if df.empty:
    st.warning("No data matches current filters.")
    st.stop()

# ── Search bar ────────────────────────────────────────────────────────────────
search = st.text_input("🔎 Search across all columns", placeholder="Type product name, city, customer…")
if search:
    mask = df.apply(lambda col: col.astype(str).str.contains(search, case=False, na=False)).any(axis=1)
    display_df = df[mask]
else:
    display_df = df

st.caption(f"Showing **{len(display_df):,}** records")

# ── Column selector ────────────────────────────────────────────────────────────
all_cols     = display_df.columns.tolist()
default_cols = ["Order ID","Order Date","Customer Name","Segment",
                "Region","Category","Sub-Category","Product Name","Sales"]
default_cols = [c for c in default_cols if c in all_cols]
sel_cols = st.multiselect("Visible columns", all_cols, default=default_cols)
if not sel_cols:
    sel_cols = default_cols

st.dataframe(
    display_df[sel_cols].reset_index(drop=True),
    use_container_width=True,
    height=420,
)

# ── Download ──────────────────────────────────────────────────────────────────
col_dl1, col_dl2 = st.columns([1, 4])
with col_dl1:
    st.download_button(
        "⬇ Download CSV",
        display_df[sel_cols].to_csv(index=False).encode(),
        "filtered_sales_data.csv",
        "text/csv",
        use_container_width=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ── Summary statistics ─────────────────────────────────────────────────────────
with st.expander("📊 Summary Statistics", expanded=False):
    st.dataframe(display_df["Sales"].describe().rename("Sales").to_frame(),
                 use_container_width=True)

# ── Top customers ─────────────────────────────────────────────────────────────
section("🏅 Top 10 Customers by Sales")
if "Customer Name" in display_df.columns:
    top_cust = (
        display_df.groupby("Customer Name")["Sales"]
        .sum().nlargest(10).reset_index().sort_values("Sales")
    )
    fig_cust = px.bar(
        top_cust, x="Sales", y="Customer Name", orientation="h",
        color="Sales", color_continuous_scale="Viridis",
        labels={"Sales":"Sales (₹)"},
    )
    fig_cust.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
        margin=dict(l=0,r=0,t=0,b=0), height=340,
    )
    st.plotly_chart(fig_cust, use_container_width=True)

# ── Customer LTV distribution ──────────────────────────────────────────────────
section("👤 Customer Lifetime Value (LTV) Distribution")
if "Customer ID" in display_df.columns:
    clv = display_df.groupby("Customer ID")["Sales"].sum().reset_index()
    clv.columns = ["Customer ID","LTV"]
    fig_ltv = px.histogram(
        clv, x="LTV", nbins=40, color_discrete_sequence=["#6C63FF"],
        labels={"LTV":"Total Sales per Customer (₹)"},
        marginal="box",
    )
    fig_ltv.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=0,b=0), height=320,
    )
    st.plotly_chart(fig_ltv, use_container_width=True)

# ── Sales anomalies highlight ──────────────────────────────────────────────────
section("🚨 Sales Anomaly Highlights (IQR Method)")
q1, q3 = display_df["Sales"].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
anom = display_df[(display_df["Sales"] < lo) | (display_df["Sales"] > hi)]

col_a, col_b = st.columns(2)
col_a.metric("Anomalous Records", f"{len(anom):,}")
col_b.metric("Anomaly Rate",      f"{len(anom)/max(len(display_df),1)*100:.1f}%")

if not anom.empty:
    anom_cols = [c for c in ["Order ID","Customer Name","Category","Sales"] if c in anom.columns]
    st.dataframe(anom[anom_cols].sort_values("Sales", ascending=False).head(50)
                  .reset_index(drop=True), use_container_width=True, height=260)

    fig_box = px.box(display_df, y="Sales", color_discrete_sequence=["#6C63FF"],
                     labels={"Sales":"Sales (₹)"}, points="outliers")
    fig_box.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0,r=0,t=0,b=0), height=260)
    st.plotly_chart(fig_box, use_container_width=True)

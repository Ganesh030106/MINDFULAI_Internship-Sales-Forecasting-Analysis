"""
pages/1_Sales_Analytics.py — Deep-dive sales breakdowns
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils import load_raw, sidebar_filters, inject_css, section, fmt_k, PALETTE

st.set_page_config(page_title="Sales Analytics", page_icon="📈", layout="wide")
inject_css()

df_raw = load_raw()
df     = sidebar_filters(df_raw)

st.markdown("## 📈 Sales Analytics")
st.caption("Deep-dive into sales trends across time, geography, segment, and product lines.")

if df.empty:
    st.warning("No data matches the current filters. Please adjust the sidebar.")
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📅 Time Trends", "🗺 Geography", "📦 Products", "👥 Segments"])

# ─────────────────────────── TAB 1: Time Trends ──────────────────────────────
with tab1:
    # Monthly trend with 3-month rolling average
    monthly = (
        df.groupby(df["Order Date"].dt.to_period("M"))["Sales"]
        .sum()
        .reset_index()
    )
    monthly["Order Date"] = monthly["Order Date"].dt.to_timestamp()
    monthly["Rolling 3M"] = monthly["Sales"].rolling(3, min_periods=1).mean()

    section("Monthly Sales vs 3-Month Rolling Average")
    fig_m = go.Figure()
    fig_m.add_trace(go.Bar(
        x=monthly["Order Date"], y=monthly["Sales"],
        name="Monthly Sales", marker_color="rgba(108,99,255,0.7)",
    ))
    fig_m.add_trace(go.Scatter(
        x=monthly["Order Date"], y=monthly["Rolling 3M"],
        name="3M Avg", line=dict(color="#FF6F61", width=2.5, dash="dot"),
    ))
    fig_m.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=0, b=0), height=340,
    )
    st.plotly_chart(fig_m, use_container_width=True)

    # Day-of-week heatmap
    section("Sales by Day of Week × Month")
    df["Weekday"] = df["Order Date"].dt.day_name()
    dow_month = (
        df.groupby(["Month", "Weekday"])["Sales"].sum()
        .reset_index()
    )
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    fig_dow = px.density_heatmap(
        dow_month, x="Month", y="Weekday", z="Sales",
        category_orders={"Weekday": day_order},
        color_continuous_scale="Plasma", text_auto=".2s",
    )
    fig_dow.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=0,b=0), height=300,
        xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig_dow, use_container_width=True)

    # Cumulative sales
    section("Cumulative Sales Growth")
    cum = monthly[["Order Date","Sales"]].copy()
    cum["Cumulative"] = cum["Sales"].cumsum()
    fig_cum = px.area(
        cum, x="Order Date", y="Cumulative",
        labels={"Cumulative": "Cumulative Sales (₹)"},
        color_discrete_sequence=["#00C9A7"],
    )
    fig_cum.update_traces(fill="tozeroy", fillcolor="rgba(0,201,167,0.15)")
    fig_cum.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=0,b=0), height=280,
    )
    st.plotly_chart(fig_cum, use_container_width=True)

# ─────────────────────────── TAB 2: Geography ────────────────────────────────
with tab2:
    section("State-Level Sales Choropleth (US)")
    state_df = df.groupby("State")["Sales"].sum().reset_index()
    fig_map = px.choropleth(
        state_df,
        locations="State", locationmode="USA-states",
        color="Sales", scope="usa",
        color_continuous_scale="Purples",
        labels={"Sales": "Sales (₹)"},
        hover_data={"Sales": ":,.0f"},
    )
    fig_map.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        geo_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=0,b=0), height=420,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Top 10 cities
    section("Top 10 Cities by Sales")
    top_cities = (
        df.groupby("City")["Sales"].sum()
        .nlargest(10)
        .reset_index()
        .sort_values("Sales")
    )
    fig_city = px.bar(
        top_cities, x="Sales", y="City", orientation="h",
        color="Sales", color_continuous_scale="Teal",
        labels={"Sales":"Sales (₹)"},
    )
    fig_city.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
        margin=dict(l=0,r=0,t=0,b=0), height=340,
    )
    st.plotly_chart(fig_city, use_container_width=True)

# ─────────────────────────── TAB 3: Products ────────────────────────────────
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        section("Top 10 Products by Sales")
        if "Product Name" in df.columns:
            top_p = (
                df.groupby("Product Name")["Sales"].sum()
                .nlargest(10).reset_index().sort_values("Sales")
            )
            fig_prod = px.bar(
                top_p, x="Sales", y="Product Name", orientation="h",
                color="Sales", color_continuous_scale="Viridis",
            )
            fig_prod.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
                margin=dict(l=0,r=0,t=0,b=0), height=380,
            )
            st.plotly_chart(fig_prod, use_container_width=True)

    with col2:
        section("Sub-Category Sales Sunburst")
        if "Sub-Category" in df.columns:
            sun_df = df.groupby(["Category","Sub-Category"])["Sales"].sum().reset_index()
            fig_sun = px.sunburst(
                sun_df, path=["Category","Sub-Category"], values="Sales",
                color="Sales", color_continuous_scale="Purples",
            )
            fig_sun.update_layout(
                margin=dict(l=0,r=0,t=0,b=0), height=380,
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_sun, use_container_width=True)

    section("Sales Distribution per Sub-Category (Box)")
    if "Sub-Category" in df.columns:
        fig_box = px.box(
            df, x="Sub-Category", y="Sales", color="Category",
            color_discrete_sequence=PALETTE,
            labels={"Sales":"Sales (₹)"},
        )
        fig_box.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=0,b=0), height=360,
            xaxis_tickangle=30,
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ─────────────────────────── TAB 4: Segments ────────────────────────────────
with tab4:
    section("Sales by Segment Over Time (Stacked Area)")
    seg_monthly = (
        df.groupby([df["Order Date"].dt.to_period("M"), "Segment"])["Sales"]
        .sum()
        .reset_index()
    )
    seg_monthly["Order Date"] = seg_monthly["Order Date"].dt.to_timestamp()
    fig_seg = px.area(
        seg_monthly, x="Order Date", y="Sales", color="Segment",
        color_discrete_sequence=PALETTE,
        labels={"Sales":"Sales (₹)"},
    )
    fig_seg.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
        margin=dict(l=0,r=0,t=0,b=0), height=340,
    )
    st.plotly_chart(fig_seg, use_container_width=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        section("Segment Share")
        seg_share = df.groupby("Segment")["Sales"].sum().reset_index()
        fig_ss = px.pie(
            seg_share, names="Segment", values="Sales", hole=0.6,
            color_discrete_sequence=PALETTE,
        )
        fig_ss.update_traces(textinfo="percent+label")
        fig_ss.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=0,b=0),
            height=300, showlegend=False,
        )
        st.plotly_chart(fig_ss, use_container_width=True)

    with col_s2:
        section("Ship Mode Mix")
        if "Ship Mode" in df.columns:
            sm = df.groupby("Ship Mode")["Sales"].sum().reset_index()
            fig_sm = px.bar(
                sm, x="Ship Mode", y="Sales",
                color="Ship Mode", color_discrete_sequence=PALETTE,
            )
            fig_sm.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                margin=dict(l=0,r=0,t=0,b=0), height=300,
            )
            st.plotly_chart(fig_sm, use_container_width=True)

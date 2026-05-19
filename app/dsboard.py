import os
import streamlit as st
st.set_page_config(page_title="Superstore Sales Dashboard", layout="wide")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import numpy as np
import plotly.express as px

# Resolve project root (one level up from app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.markdown("<h1 style='text-align: center; color: white;'>SUPERSTORE SALES DASHBOARD</h1>", unsafe_allow_html=True)

# Rupee formatting utility
def rupees(x):
    return f"₹{x:,.0f}"


@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "Cleaned_dataset.csv"))
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    return df.dropna(subset=['Order Date', 'Sales'])


df = load_data()

# --- Sidebar Filters ---
regions = df['Region'].unique()
categories = df['Category'].unique()
selected_region = st.sidebar.selectbox("Select Region", options=["All"] + sorted(map(str, regions)))
selected_category = st.sidebar.selectbox("Select Category", options=["All"] + sorted(map(str, categories)))

filtered_df = df.copy()
if selected_region != "All":
    filtered_df = filtered_df[filtered_df['Region'].astype(str) == selected_region]
if selected_category != "All":
    filtered_df = filtered_df[filtered_df['Category'].astype(str) == selected_category]

# --- KPI Cards ---
col1, col2, col3, col4 = st.columns(4)
total_sales = filtered_df['Sales'].sum()
order_count = filtered_df['Order ID'].nunique() if 'Order ID' in filtered_df else len(filtered_df)
avg_sales = filtered_df['Sales'].mean()
max_sales = filtered_df['Sales'].max()

col1.metric("Total Sales", rupees(total_sales))
col2.metric("Order Count", f"{order_count:,}")
col3.metric("Average Sale", rupees(avg_sales))
col4.metric("Max Sale", rupees(max_sales))

st.markdown("---")

# --- Sales Time Series ---
st.subheader("📈 Sales Over Time")
df_grouped = filtered_df.groupby('Order Date')['Sales'].sum().reset_index().rename(columns={'Order Date': 'ds', 'Sales': 'y'})
st.line_chart(df_grouped.set_index("ds")["y"])

# --- Prophet Forecast ---
if len(df_grouped) > 30:
    st.subheader("🔮 30-Day Sales Forecast")
    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(df_grouped)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    fig1 = model.plot(forecast)
    plt.title("Forecast: Next 30 Days (₹)")
    st.pyplot(fig1)
else:
    st.info("Not enough data for forecasting with current filter.")

# --- Monthly Sales Trend ---
st.subheader("📅 Monthly Sales Trend")
monthly_sales = filtered_df.resample("ME", on="Order Date")["Sales"].sum().reset_index()
fig2, ax2 = plt.subplots()
ax2.plot(monthly_sales["Order Date"], monthly_sales["Sales"], marker='o', color='teal')
ax2.set_xlabel("Month")
ax2.set_ylabel("Sales (₹)")
ax2.set_title("Monthly Sales Over Time")
st.pyplot(fig2)

# --- Sales by Category ---
st.subheader("Sales by Category")
category_sales = filtered_df.groupby("Category")["Sales"].sum().sort_values()
fig3, ax3 = plt.subplots()
category_sales.plot(kind='barh', ax=ax3, color="orange")
ax3.set_xlabel("Sales (₹)")
ax3.set_title("Total Sales by Category")
st.pyplot(fig3)

# --- Sales by Region ---
st.subheader("Sales by Region")
region_sales = filtered_df.groupby("Region")["Sales"].sum().sort_values()
fig4, ax4 = plt.subplots()
region_sales.plot(kind='barh', ax=ax4, color="royalblue")
ax4.set_xlabel("Sales (₹)")
ax4.set_title("Total Sales by Region")
st.pyplot(fig4)

# --- Top 10 Products by Sales ---
if "Product Name" in filtered_df.columns:
    st.subheader("🏆 Top 10 Products by Sales")
    top_products = (
        filtered_df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10)
    )
    fig5, ax5 = plt.subplots()
    top_products.plot(kind="bar", ax=ax5, color="green")
    ax5.set_ylabel("Sales (₹)")
    ax5.set_title("Top 10 Products by Sales")
    st.pyplot(fig5)

# --- Sales Distribution Histogram ---
st.subheader("Sales Value Distribution")
fig6, ax6 = plt.subplots()
filtered_df["Sales"].plot(kind="hist", bins=30, alpha=0.7, color="navy", edgecolor="white", ax=ax6)
ax6.set_xlabel("Sales (₹)")
ax6.set_title("Sales Distribution")
st.pyplot(fig6)


# --- Sunburst Chart: Category > Sub-Category (Requires Plotly) ---
if "Sub-Category" in filtered_df.columns:
    st.subheader("Category & Sub-Category Sales Breakdown")
    sunburst_df = filtered_df.groupby(['Category', 'Sub-Category'])['Sales'].sum().reset_index()
    fig8 = px.sunburst(sunburst_df, path=['Category', 'Sub-Category'], values='Sales',
                      color='Sales', color_continuous_scale='Blues',
                      labels={'Sales':'Sales (₹)'})
    st.plotly_chart(fig8, use_container_width=True)

# --- Customer Lifetime Value Histogram ---
if "Customer ID" in filtered_df.columns:
    st.subheader("Customer Lifetime Value (LTV) Distribution")
    clv = filtered_df.groupby('Customer ID')['Sales'].sum()
    fig9, ax9 = plt.subplots()
    sns.histplot(clv, bins=30, kde=True, ax=ax9, color='teal')
    ax9.set_xlabel('Total Sales per Customer (₹)')
    ax9.set_title('Customer Lifetime Value Distribution')
    st.pyplot(fig9)

# --- Month-over-Month Sales Delta ---
this_month = filtered_df[filtered_df['Order Date'].dt.month == filtered_df['Order Date'].max().month]
last_month = filtered_df[filtered_df['Order Date'].dt.month == (filtered_df['Order Date'].max().month - 1)]
this_sales = this_month['Sales'].sum()
last_sales = last_month['Sales'].sum() if not last_month.empty else 0
delta = this_sales - last_sales
st.metric("Sales This Month", rupees(this_sales), delta=(f"{rupees(abs(delta))} {'↑' if delta > 0 else '↓'}"))

if 'Ship Mode' in filtered_df.columns:
    st.subheader("🚚 Sales Distribution by Ship Mode")
    shipmode_sales = filtered_df.groupby('Ship Mode')['Sales'].sum()
    fig_ship, ax_ship = plt.subplots()
    ax_ship.pie(shipmode_sales, labels=shipmode_sales.index, autopct=lambda p: f"{p:.1f}%", startangle=140, colors=sns.color_palette("Set2"))
    ax_ship.axis('equal')
    st.pyplot(fig_ship)

if 'Sub-Category' in filtered_df.columns:
    st.subheader("📦 Sales Distribution per Sub-Category")
    fig_box, ax_box = plt.subplots(figsize=(10,5))
    sns.boxplot(data=filtered_df,x="Sub-Category",y="Sales",hue="Sub-Category",ax=ax_box,palette="Set3",legend=False)
    ax_box.set_xlabel("Sub-Category")
    ax_box.set_ylabel("Sales (₹)")
    ax_box.set_title("Sub-Category Sales Boxplot")
    plt.xticks(rotation=45)
    st.pyplot(fig_box)

st.subheader("📊 Cumulative Sales Over Time")
cum_sales = df_grouped.copy()
cum_sales['Cumulative Sales'] = cum_sales['y'].cumsum()
fig_cum, ax_cum = plt.subplots()
ax_cum.plot(cum_sales['ds'], cum_sales['Cumulative Sales'], color='indigo')
ax_cum.set_xlabel("Date")
ax_cum.set_ylabel("Cumulative Sales (₹)")
ax_cum.set_title("Cumulative Sales Growth")
st.pyplot(fig_cum)

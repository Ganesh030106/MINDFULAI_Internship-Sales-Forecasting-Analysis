import os
import pandas as pd
import numpy as np
import streamlit as st

# ── Resolve project root (works from any CWD) ──────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Colour palette ─────────────────────────────────────────────────────────
PALETTE = ["#6C63FF", "#FF6F61", "#00C9A7", "#FFD166", "#4CC9F0", "#F72585"]
BG_CARD  = "rgba(255,255,255,0.04)"

# ── Data loader ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_raw():
    """Load the original train.csv (all original columns preserved)."""
    path = os.path.join(BASE_DIR, "data", "train.csv")
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  errors="coerce")
    df["Sales"]      = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"])
    # Derived time columns
    df["Year"]    = df["Order Date"].dt.year
    df["Month"]   = df["Order Date"].dt.to_period("M").astype(str)
    df["Quarter"] = df["Order Date"].dt.to_period("Q").astype(str)
    return df

# ── Formatting helpers ─────────────────────────────────────────────────────
def fmt_inr(x: float) -> str:
    return f"₹{x:,.0f}"

def fmt_k(x: float) -> str:
    if abs(x) >= 1_000_000:
        return f"₹{x/1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"₹{x/1_000:.1f}K"
    return fmt_inr(x)

# ── Sidebar filter ─────────────────────────────────────────────────────────
def sidebar_filters(df: pd.DataFrame):
    st.sidebar.markdown(
        "<div style='text-align:center;padding:12px 0 4px'>"
        "<span style='font-size:2rem'>📊</span><br>"
        "<b style='font-size:1.1rem;color:#6C63FF'>MINDFULAI</b><br>"
        "<small style='color:#888'>Sales Intelligence</small>"
        "</div><hr style='border-color:#333'>",
        unsafe_allow_html=True,
    )

    # Date range
    min_d, max_d = df["Order Date"].min().date(), df["Order Date"].max().date()
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = date_range
    else:
        start, end = min_d, max_d

    # Dimension filters
    regions    = sorted(df["Region"].dropna().unique())
    categories = sorted(df["Category"].dropna().unique())
    segments   = sorted(df["Segment"].dropna().unique())

    sel_region   = st.sidebar.multiselect("🌐 Region",   regions,    default=regions)
    sel_category = st.sidebar.multiselect("📦 Category", categories, default=categories)
    sel_segment  = st.sidebar.multiselect("👥 Segment",  segments,   default=segments)

    # Apply filters
    mask = (
        (df["Order Date"].dt.date >= start) &
        (df["Order Date"].dt.date <= end)   &
        (df["Region"].isin(sel_region))     &
        (df["Category"].isin(sel_category)) &
        (df["Segment"].isin(sel_segment))
    )
    filtered = df[mask].copy()

    st.sidebar.markdown(
        f"<small style='color:#888'>Showing <b style='color:#6C63FF'>"
        f"{len(filtered):,}</b> of {len(df):,} records</small>",
        unsafe_allow_html=True,
    )
    return filtered

# ── Shared CSS injector ────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: rgba(108,99,255,0.08);
        border: 1px solid rgba(108,99,255,0.25);
        border-radius: 12px;
        padding: 18px 20px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(108,99,255,0.25);
    }
    [data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #aaa !important; font-size: 0.8rem; }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.15rem; font-weight: 600; color: #fff;
        border-left: 4px solid #6C63FF; padding-left: 12px;
        margin: 24px 0 16px;
    }

    /* ── Divider ── */
    hr { border-color: rgba(255,255,255,0.08) !important; margin: 24px 0 !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background: #0d0d1a !important; }
    [data-testid="stSidebar"] hr { border-color: #222 !important; }

    /* ── Tabs ── */
    [data-testid="stTabs"] button { font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

def section(title: str):
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)

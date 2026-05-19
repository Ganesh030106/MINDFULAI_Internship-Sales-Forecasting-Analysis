"""
pages/3_ML_Insights.py — Live XGBoost classification + Gradient Boosting regression
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBClassifier
from utils import load_raw, sidebar_filters, inject_css, section, fmt_k, PALETTE

st.set_page_config(page_title="ML Insights", page_icon="🤖", layout="wide")
inject_css()

df_raw = load_raw()
df     = sidebar_filters(df_raw)

st.markdown("## 🤖 ML Insights")
st.caption("Live XGBoost classification & Gradient Boosting regression — trained on your filtered data.")

if df.empty:
    st.warning("No data matches current filters.")
    st.stop()

# ── Feature engineering ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Training ML models…")
def train_models(_df):
    d = _df.copy()
    d["Order Date"] = pd.to_datetime(d["Order Date"], errors="coerce")
    d["Order_Month"]   = d["Order Date"].dt.month
    d["Order_Quarter"] = d["Order Date"].dt.quarter
    d["Order_Weekday"] = d["Order Date"].dt.weekday

    cat_cols = ["Category","Sub-Category","Region","Segment","Ship Mode"]
    le = LabelEncoder()
    for col in cat_cols:
        if col in d.columns:
            d[col] = le.fit_transform(d[col].astype(str))

    d["High_Sales"] = (d["Sales"] > d["Sales"].median()).astype(int)
    features = [c for c in cat_cols if c in d.columns] + \
               ["Order_Month","Order_Quarter","Order_Weekday"]
    d = d[features + ["Sales","High_Sales"]].dropna()

    # --- Classification ---
    X, y = d[features], d["High_Sales"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y,
                                           test_size=0.15, random_state=42)
    clf = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.08,
                        use_label_encoder=False, eval_metric="logloss",
                        random_state=42)
    clf.fit(Xtr, ytr)
    y_pred  = clf.predict(Xte)
    y_proba = clf.predict_proba(Xte)[:, 1]

    # --- Regression ---
    Xr, yr_ = d[features], d["Sales"]
    Xrtr, Xrte, yrtr, yrte = train_test_split(Xr, yr_, test_size=0.15, random_state=42)
    reg = GradientBoostingRegressor(n_estimators=120, max_depth=4,
                                    learning_rate=0.1, random_state=42)
    reg.fit(Xrtr, yrtr)
    y_rpred = reg.predict(Xrte)

    return clf, reg, features, Xte, yte, y_pred, y_proba, Xrte, yrte, y_rpred

with st.spinner("Training models on filtered data…"):
    clf, reg, features, Xte, yte, y_pred, y_proba, Xrte, yrte, y_rpred = train_models(df)

tab_cls, tab_reg = st.tabs(["🔍 Classification", "📐 Regression"])

# ─────────────────────── CLASSIFICATION ─────────────────────────────────────
with tab_cls:
    acc = accuracy_score(yte, y_pred)
    fpr, tpr, _ = roc_curve(yte, y_proba)
    roc_auc = auc(fpr, tpr)

    m1, m2, m3 = st.columns(3)
    m1.metric("Test Accuracy",   f"{acc:.1%}")
    m2.metric("ROC-AUC",         f"{roc_auc:.3f}")
    m3.metric("Test Samples",    f"{len(yte):,}")

    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("Confusion Matrix")
        cm = confusion_matrix(yte, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True, aspect="auto",
            color_continuous_scale="Purples",
            labels=dict(x="Predicted", y="Actual"),
            x=["Low Sales","High Sales"], y=["Low Sales","High Sales"],
        )
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=0,r=0,t=0,b=0), height=300,
                              coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        section("ROC Curve")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"AUC = {roc_auc:.3f}",
                                     line=dict(color="#6C63FF", width=2.5)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     name="Random", line=dict(color="#555", dash="dash")))
        fig_roc.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=0,b=0), height=300,
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    section("Feature Importance (XGBoost)")
    imp = pd.DataFrame({"Feature": features,
                         "Importance": clf.feature_importances_}).sort_values("Importance")
    fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Purples")
    fig_imp.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
                          margin=dict(l=0,r=0,t=0,b=0), height=300)
    st.plotly_chart(fig_imp, use_container_width=True)

# ─────────────────────── REGRESSION ─────────────────────────────────────────
with tab_reg:
    rmse = np.sqrt(mean_squared_error(yrte, y_rpred))
    mae  = mean_absolute_error(yrte, y_rpred)
    r2   = r2_score(yrte, y_rpred)

    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score",  f"{r2:.3f}")
    m2.metric("RMSE",      fmt_k(rmse))
    m3.metric("MAE",       fmt_k(mae))

    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("Actual vs Predicted Sales")
        scatter_df = pd.DataFrame({"Actual": yrte.values, "Predicted": y_rpred})
        fig_sc = px.scatter(scatter_df, x="Actual", y="Predicted",
                            opacity=0.55, color_discrete_sequence=["#6C63FF"])
        min_v = min(scatter_df["Actual"].min(), scatter_df["Predicted"].min())
        max_v = max(scatter_df["Actual"].max(), scatter_df["Predicted"].max())
        fig_sc.add_trace(go.Scatter(x=[min_v,max_v], y=[min_v,max_v],
                                    mode="lines", name="Perfect Fit",
                                    line=dict(color="#FF6F61", dash="dash")))
        fig_sc.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)",
                             margin=dict(l=0,r=0,t=0,b=0), height=320)
        st.plotly_chart(fig_sc, use_container_width=True)

    with col2:
        section("Residuals Distribution")
        residuals = yrte.values - y_rpred
        fig_res = px.histogram(residuals, nbins=40, color_discrete_sequence=["#00C9A7"],
                               labels={"value":"Residual (Actual − Predicted)"})
        fig_res.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                              margin=dict(l=0,r=0,t=0,b=0), height=320)
        st.plotly_chart(fig_res, use_container_width=True)

    section("Feature Importance (Gradient Boosting)")
    rimp = pd.DataFrame({"Feature": features,
                          "Importance": reg.feature_importances_}).sort_values("Importance")
    fig_rimp = px.bar(rimp, x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale="Teal")
    fig_rimp.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
                           margin=dict(l=0,r=0,t=0,b=0), height=300)
    st.plotly_chart(fig_rimp, use_container_width=True)

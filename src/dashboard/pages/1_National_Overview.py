"""
Page 1 — National Overview

Shows national-level data from India's digital transformation analysis:
wireless subscriber growth, structural breaks, provider market share,
digital transactions, and market concentration (HHI).
"""

import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.analysis.obj1_wireless_growth import (
    get_national_wireless_ts,
    detect_structural_breaks,
    compute_hhi,
    get_provider_shares,
)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"

SQLITE_PATH = ROOT / "db" / "sqlite" / "dsm.db"


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_wireless_ts() -> pd.DataFrame:
    client = MongoClient("mongodb://localhost:27017")
    db = client["dsm"]
    return get_national_wireless_ts(db)


@st.cache_data(ttl=3600)
def load_structural_breaks(series_values: list, n_bkps: int = 2) -> list:
    return detect_structural_breaks(np.array(series_values), n_bkps=n_bkps)


@st.cache_data(ttl=3600)
def load_hhi() -> pd.DataFrame:
    client = MongoClient("mongodb://localhost:27017")
    db = client["dsm"]
    return compute_hhi(db)


@st.cache_data(ttl=3600)
def load_provider_shares() -> pd.DataFrame:
    client = MongoClient("mongodb://localhost:27017")
    db = client["dsm"]
    return get_provider_shares(db)


@st.cache_data(ttl=3600)
def load_digital_transactions() -> pd.DataFrame:
    conn = sqlite3.connect(str(SQLITE_PATH))
    df = pd.read_sql_query("SELECT * FROM digital_transactions", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.header("📊 National Overview")

# Load data
ts = load_wireless_ts()
break_indices = load_structural_breaks(ts["total_wireless"].tolist(), n_bkps=2)
break_dates = [ts["date"].iloc[min(idx, len(ts) - 1)] for idx in break_indices]
hhi_df = load_hhi()
shares = load_provider_shares()
digital_txn = load_digital_transactions()

# -----------------------------------------------------------------------
# Row 1 — Key metrics
# -----------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    latest_wireless = ts["total_wireless"].iloc[-1]
    st.metric(
        label="Total Wireless Subscribers",
        value=f"{latest_wireless / 1e6:,.1f} M",
    )

with col2:
    break_labels = ", ".join(d.strftime("%b %Y") for d in break_dates)
    st.metric(
        label="Structural Breaks Detected",
        value="2",
        delta=break_labels,
    )

with col3:
    upi_latest = (
        digital_txn["bhim_txn_crores"].iloc[-1]
        / digital_txn["digital_txn_crores"].iloc[-1]
        * 100
    )
    upi_earliest = (
        digital_txn["bhim_txn_crores"].iloc[0]
        / digital_txn["digital_txn_crores"].iloc[0]
        * 100
    )
    st.metric(
        label="UPI Share (latest)",
        value=f"{upi_latest:.1f}%",
        delta=f"{upi_latest - upi_earliest:+.1f} pp from earliest",
    )

# -----------------------------------------------------------------------
# Row 2 — Wireless growth + provider market share
# -----------------------------------------------------------------------
left2, right2 = st.columns(2)

with left2:
    st.subheader("Wireless Subscriber Growth")
    fig_wireless = px.line(
        ts,
        x="date",
        y="total_wireless",
        labels={"date": "Date", "total_wireless": "Total Wireless Subscribers"},
    )
    fig_wireless.update_traces(line_color=BLUE)
    # Structural break annotations
    fig_wireless.add_vline(
        x=pd.Timestamp("2011-01-01"),
        line_dash="dash",
        line_color=ORANGE,
        annotation_text="Mobile Explosion",
        annotation_position="top left",
    )
    fig_wireless.add_vline(
        x=pd.Timestamp("2016-06-01"),
        line_dash="dash",
        line_color=RED,
        annotation_text="Jio Entry",
        annotation_position="top left",
    )
    fig_wireless.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_wireless, use_container_width=True)

with right2:
    st.subheader("Provider Market Share")
    fig_share = go.Figure()
    for col_name in shares.columns:
        fig_share.add_trace(
            go.Scatter(
                x=shares.index,
                y=shares[col_name],
                name=col_name,
                mode="lines",
                stackgroup="one",
            )
        )
    fig_share.update_layout(
        yaxis_title="Market Share (%)",
        xaxis_title="Year",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig_share, use_container_width=True)

# -----------------------------------------------------------------------
# Row 3 — Digital transactions + payment composition
# -----------------------------------------------------------------------
left3, right3 = st.columns(2)

with left3:
    st.subheader("Digital Transaction Volume")
    fig_digital = px.line(
        digital_txn,
        x="date",
        y="digital_txn_crores",
        labels={"date": "Date", "digital_txn_crores": "Digital Txn (Crores)"},
    )
    fig_digital.update_traces(line_color=GREEN)
    fig_digital.add_vline(
        x=pd.Timestamp("2020-03-01"),
        line_dash="dash",
        line_color=RED,
        annotation_text="COVID-19",
        annotation_position="top left",
    )
    fig_digital.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_digital, use_container_width=True)

with right3:
    st.subheader("Payment Composition")
    txn = digital_txn.copy()
    txn["upi_share"] = txn["bhim_txn_crores"] / txn["digital_txn_crores"] * 100
    txn["debit_share"] = txn["debit_card_crores"] / txn["digital_txn_crores"] * 100
    txn["other_share"] = 100 - txn["upi_share"] - txn["debit_share"]

    fig_payment = go.Figure()
    for col_name, label in [
        ("upi_share", "UPI/BHIM"),
        ("debit_share", "Debit Card"),
        ("other_share", "Other Digital"),
    ]:
        fig_payment.add_trace(
            go.Scatter(
                x=txn["date"],
                y=txn[col_name],
                name=label,
                mode="lines",
                stackgroup="one",
            )
        )
    fig_payment.update_layout(
        yaxis_title="Share (%)",
        xaxis_title="Date",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig_payment, use_container_width=True)

# -----------------------------------------------------------------------
# Row 4 — HHI over time (full width)
# -----------------------------------------------------------------------
st.subheader("Market Concentration (HHI) Over Time")

national_hhi = hhi_df.groupby("year")["hhi"].mean().reset_index()

fig_hhi = px.line(
    national_hhi,
    x="year",
    y="hhi",
    labels={"year": "Year", "hhi": "HHI"},
)
fig_hhi.update_traces(line_color=BLUE)

# Threshold reference lines
fig_hhi.add_hline(
    y=1500,
    line_dash="dot",
    line_color=ORANGE,
    annotation_text="Moderately concentrated",
    annotation_position="top right",
)
fig_hhi.add_hline(
    y=2500,
    line_dash="dot",
    line_color=RED,
    annotation_text="Highly concentrated",
    annotation_position="top right",
)
fig_hhi.update_layout(margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig_hhi, use_container_width=True)

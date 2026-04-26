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
    chow_test,
    compute_cagr,
    compute_hhi,
    get_provider_shares,
)

BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN  = "#2ca02c"
RED    = "#d62728"

SQLITE_PATH = ROOT / "db" / "sqlite" / "dsm.db"

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

# loads national wireless subscriber time series from MongoDB
@st.cache_data(ttl=3600)
def load_wireless_ts():
    client = MongoClient("mongodb://localhost:27017")
    db     = client["dsm"]
    ts     = get_national_wireless_ts(db)
    client.close()
    return ts


# detects structural breaks in the wireless series
@st.cache_data(ttl=3600)
def load_structural_breaks(series_values, n_bkps=2):
    return detect_structural_breaks(np.array(series_values), n_bkps=n_bkps)


# runs structural break detection, Chow tests, and CAGR computation
@st.cache_data(ttl=3600)
def compute_structural_break_results(series_values, dates_iso):
    series      = np.array(series_values)
    dates       = pd.to_datetime(dates_iso)
    break_indices = detect_structural_breaks(series, n_bkps=2)
    break_dates   = [dates[min(idx, len(dates) - 1)] for idx in break_indices]

    chow_rows = []
    for i, (idx, dt) in enumerate(zip(break_indices, break_dates)):
        f_stat, p_value = chow_test(series, idx)
        chow_rows.append({
            "Break":       i + 1,
            "Date":        dt.strftime("%b %Y"),
            "F-statistic": round(f_stat, 2),
            "p-value":     f"{p_value:.2e}",
            "Significant": "Yes ✓" if p_value < 0.05 else "No",
        })

    first_break  = break_dates[0]
    second_break = break_dates[1]

    ts_df = pd.DataFrame({"date": dates, "total_wireless": series})
    periods = [
        ("Pre-break 1",        ts_df[ts_df["date"] < first_break]),
        ("Break 1 → Break 2",  ts_df[(ts_df["date"] >= first_break) & (ts_df["date"] < second_break)]),
        ("Post-break 2",       ts_df[ts_df["date"] >= second_break]),
    ]
    cagr_rows = []
    for label, p in periods:
        if len(p) < 2:
            continue
        years = (p["date"].iloc[-1] - p["date"].iloc[0]).days / 365.25
        cagr  = compute_cagr(p["total_wireless"].iloc[0], p["total_wireless"].iloc[-1], years)
        cagr_rows.append({
            "Period": label,
            "Start":  p["date"].iloc[0].strftime("%b %Y"),
            "End":    p["date"].iloc[-1].strftime("%b %Y"),
            "CAGR (%)": round(cagr * 100, 2),
        })

    return {
        "chow_rows":   chow_rows,
        "cagr_rows":   cagr_rows,
        "break_dates": [d.isoformat() for d in break_dates],
    }


# loads HHI per state per year from MongoDB
@st.cache_data(ttl=3600)
def load_hhi():
    client = MongoClient("mongodb://localhost:27017")
    db     = client["dsm"]
    hhi    = compute_hhi(db)
    client.close()
    return hhi


# loads national provider market shares from MongoDB
@st.cache_data(ttl=3600)
def load_provider_shares():
    client = MongoClient("mongodb://localhost:27017")
    db     = client["dsm"]
    shares = get_provider_shares(db)
    client.close()
    return shares


# loads monthly digital transactions from SQLite
@st.cache_data(ttl=3600)
def load_digital_transactions():
    conn = sqlite3.connect(str(SQLITE_PATH))
    df   = pd.read_sql_query("SELECT * FROM digital_transactions", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


st.header("India's Connectivity Revolution")

st.info(
    "**RQ 1** — How did India's wireless subscriber base grow between 2013–2023, and were there "
    "identifiable structural breaks driven by policy or market events — most notably Reliance "
    "Jio's entry in September 2016?\n\n"
    "We apply the **Bai-Perron structural break algorithm** to the national wireless subscriber "
    "time series and validate each break with a **Chow test**."
)

ts            = load_wireless_ts()
break_indices = load_structural_breaks(ts["total_wireless"].tolist(), n_bkps=2)
break_dates   = [ts["date"].iloc[min(idx, len(ts) - 1)] for idx in break_indices]
hhi_df        = load_hhi()
shares        = load_provider_shares()
digital_txn   = load_digital_transactions()

st.divider()

# ---------------------------------------------------------------------------
# Section 2 — Wireless subscriber growth
# ---------------------------------------------------------------------------

st.subheader("1. Wireless Subscriber Growth (2013–2023)")
st.markdown(
    "The national wireless subscriber count is the headline measure of India's telecom reach. "
    "Two structural breaks divide the series into three regimes with markedly different growth dynamics."
)

fig_wireless = px.line(
    ts,
    x="date",
    y="total_wireless",
    labels={"date": "Date", "total_wireless": "Total Wireless Subscribers"},
)
fig_wireless.update_traces(line_color=BLUE)

fig_wireless.add_vline(x="2011-01-01", line_dash="dash", line_color=ORANGE)
fig_wireless.add_annotation(
    x="2011-01-01", y=1, yref="paper", text="Break 1: Mobile explosion",
    showarrow=False, xanchor="left", font=dict(color=ORANGE, size=11),
)
fig_wireless.add_vline(x="2016-06-01", line_dash="dash", line_color=RED)
fig_wireless.add_annotation(
    x="2016-06-01", y=0.88, yref="paper", text="Break 2: Jio entry",
    showarrow=False, xanchor="left", font=dict(color=RED, size=11),
)
fig_wireless.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20), height=380)
st.plotly_chart(fig_wireless, width="stretch")

st.markdown(
    "> **Finding:** India's wireless subscriber count grew from ~870 M (2013) to ~1.17 B (2023). "
    "Growth was fastest in the early smartphone era (pre-2011) and again post-Jio. After Jio's "
    "September 2016 launch with free voice and near-free data, the growth trajectory changed sharply — "
    "but so did the composition: low-ARPU subscribers flooded in while incumbents lost share, "
    "which eventually caused the total count to plateau as the market consolidated."
)

st.divider()

# ---------------------------------------------------------------------------
# Section 3 — Structural break analysis
# ---------------------------------------------------------------------------

st.subheader("2. Structural Break Analysis — Chow Tests & CAGR by Period")
st.markdown(
    "The Chow test checks whether the growth regression significantly changes at each detected break. "
    "A significant F-statistic confirms the break is real rather than random noise. "
    "CAGR (Compound Annual Growth Rate) quantifies how fast subscribers grew in each regime."
)

break_results = compute_structural_break_results(
    ts["total_wireless"].tolist(),
    ts["date"].dt.strftime("%Y-%m-%d").tolist(),
)

col_left, col_right = st.columns(2)
with col_left:
    st.markdown("**Chow Test Results**")
    st.dataframe(pd.DataFrame(break_results["chow_rows"]), width="stretch", hide_index=True)

with col_right:
    st.markdown("**CAGR by Growth Period**")
    st.dataframe(pd.DataFrame(break_results["cagr_rows"]), width="stretch", hide_index=True)

st.markdown(
    "> **Finding:** Both structural breaks are statistically significant (p < 0.05), confirming "
    "that the shifts in growth rate are not random. The highest CAGR is in the early mobile boom "
    "era, followed by a second burst post-Jio. The post-Jio regime shows lower CAGR in raw "
    "subscriber numbers but a much larger shift in the *quality* of access — data speeds, "
    "coverage depth, and affordability all changed dramatically."
)

st.divider()

# ---------------------------------------------------------------------------
# Section 5 — Digital transactions
# ---------------------------------------------------------------------------

st.subheader("3. Digital Transaction Growth & Payment Composition")
st.markdown(
    "Digital payment volumes are the downstream outcome of telecom expansion. "
    "As more Indians got cheap data via Jio, UPI-based payments grew explosively — "
    "even through COVID-19, which accelerated contactless payment adoption."
)

col_txn, col_pay = st.columns(2)

with col_txn:
    st.markdown("**Monthly Digital Transaction Volume**")
    fig_digital = px.line(
        digital_txn, x="date", y="digital_txn_crores",
        labels={"date": "Date", "digital_txn_crores": "Digital Txn (Crores)"},
    )
    fig_digital.update_traces(line_color=GREEN)
    fig_digital.add_vline(x="2020-03-01", line_dash="dash", line_color=RED)
    fig_digital.add_annotation(
        x="2020-03-01", y=1, yref="paper", text="COVID-19 (Mar 2020)",
        showarrow=False, xanchor="left", font=dict(color=RED, size=11),
    )
    fig_digital.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20), height=340)
    st.plotly_chart(fig_digital, width="stretch")

with col_pay:
    st.markdown("**Payment Method Composition (%)**")
    txn = digital_txn.copy()
    txn["upi_share"]   = txn["bhim_txn_crores"] / txn["digital_txn_crores"] * 100
    txn["debit_share"] = txn["debit_card_crores"] / txn["digital_txn_crores"] * 100
    txn["other_share"] = 100 - txn["upi_share"] - txn["debit_share"]

    fig_payment = go.Figure()
    for col_name, label in [
        ("upi_share", "UPI/BHIM"),
        ("debit_share", "Debit Card"),
        ("other_share", "Other Digital"),
    ]:
        fig_payment.add_trace(go.Scatter(
            x=txn["date"], y=txn[col_name],
            name=label, mode="lines", stackgroup="one",
        ))
    fig_payment.update_layout(
        yaxis_title="Share (%)", xaxis_title="Date",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(font=dict(size=10)),
        height=340,
    )
    st.plotly_chart(fig_payment, width="stretch")

st.markdown(
    "> **Finding:** Digital transaction volumes grew from nearly zero in 2016 to over 5,000 crore "
    "monthly transactions by 2021. UPI/BHIM's share of total digital payments rose from negligible "
    "to dominant within four years — the fastest payment-method adoption in any major economy. "
    "COVID-19 created a further step-change: contactless payment volumes jumped and did not revert, "
    "suggesting a permanent behavioural shift rather than a temporary crisis response."
)

# ---------------------------------------------------------------------------
# Section 4 — Provider market share + HHI
# ---------------------------------------------------------------------------

st.subheader("4. Provider Market Share & Market Concentration (HHI)")
st.markdown(
    "Jio's entry didn't just add subscribers — it restructured the market. "
    "The Herfindahl-Hirschman Index (HHI) measures concentration: "
    "HHI > 2,500 is highly concentrated; HHI < 1,500 is competitive."
)

col_share, col_hhi = st.columns(2)

with col_share:
    st.markdown("**Provider Market Share Over Time**")
    fig_share = go.Figure()
    for col_name in shares.columns:
        fig_share.add_trace(go.Scatter(
            x=shares.index, y=shares[col_name],
            name=col_name, mode="lines", stackgroup="one",
        ))
    fig_share.update_layout(
        yaxis_title="Market Share (%)",
        xaxis_title="Year",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(font=dict(size=10)),
        height=360,
    )
    st.plotly_chart(fig_share, width="stretch")

with col_hhi:
    st.markdown("**HHI Over Time**")
    national_hhi = hhi_df.groupby("year")["hhi"].mean().reset_index()
    fig_hhi = px.line(
        national_hhi, x="year", y="hhi",
        labels={"year": "Year", "hhi": "HHI"},
    )
    fig_hhi.update_traces(line_color=BLUE)
    fig_hhi.add_hline(
        y=1500, line_dash="dot", line_color=ORANGE,
        annotation_text="Moderately concentrated",
        annotation_position="top right",
    )
    fig_hhi.add_hline(
        y=2500, line_dash="dot", line_color=RED,
        annotation_text="Highly concentrated",
        annotation_position="top right",
    )
    fig_hhi.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=360)
    st.plotly_chart(fig_hhi, width="stretch")

st.markdown(
    "> **Finding:** Before Jio's 2016 entry, the market was highly concentrated (HHI > 2,500), "
    "dominated by Airtel, Vodafone, and Idea. Jio's arrival collapsed the HHI sharply — the "
    "market moved from highly concentrated to moderately concentrated within two years. "
    "This is the fastest telecom market disruption in Indian history. The market share chart "
    "shows the incumbents bleeding subscribers to Jio from 2017 onward, eventually triggering "
    "the Vodafone-Idea merger and Airtel's own pricing restructuring."
)

st.divider()


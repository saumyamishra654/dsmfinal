import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pymongo import MongoClient
from statsmodels.tsa.stattools import grangercausalitytests

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.analysis.obj1_wireless_growth import get_national_wireless_ts

BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN  = "#2ca02c"
RED    = "#d62728"

SQLITE_PATH = ROOT / "db" / "sqlite" / "dsm.db"

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

# runs Granger causality test on wireless and digital transaction growth
@st.cache_data(ttl=3600)
def compute_granger_results():
    conn = sqlite3.connect(str(SQLITE_PATH))
    digital_txn = pd.read_sql_query("SELECT * FROM digital_transactions", conn)
    conn.close()

    client = MongoClient("mongodb://localhost:27017")
    db     = client["dsm"]
    wireless = get_national_wireless_ts(db)
    client.close()

    merged = pd.merge(
        digital_txn[["year", "month", "digital_txn_crores"]],
        wireless[["year", "month", "total_wireless"]],
        on=["year", "month"],
        how="inner",
    ).sort_values(["year", "month"]).reset_index(drop=True)

    merged["wireless_growth"] = merged["total_wireless"].pct_change()
    merged["txn_growth"]      = merged["digital_txn_crores"].pct_change()
    merged = merged.dropna(subset=["wireless_growth", "txn_growth"])

    data    = merged[["txn_growth", "wireless_growth"]].values
    results = grangercausalitytests(data, maxlag=4, verbose=False)

    rows = []
    for lag in range(1, 5):
        f_stat  = results[lag][0]["ssr_ftest"][0]
        p_value = results[lag][0]["ssr_ftest"][1]
        rows.append({
            "Lag (months)": lag,
            "F-statistic":  round(f_stat, 4),
            "p-value":      round(p_value, 4),
            "Significant":  "Yes ✓" if p_value < 0.05 else "No",
        })
    return pd.DataFrame(rows)


# loads and normalizes electricity, wireless, and digital transaction data
@st.cache_data(ttl=3600)
def load_corroboration_data():
    conn = sqlite3.connect(str(SQLITE_PATH))

    elec = pd.read_sql_query(
        """
        SELECT year, SUM(energy_gwh) AS energy_gwh
        FROM electricity_consumption
        WHERE sector IN ('Domestic', 'Commercial')
        GROUP BY year ORDER BY year
        """,
        conn,
    )
    digital = pd.read_sql_query(
        """
        SELECT year, AVG(digital_txn_crores) AS digital_txn_mean
        FROM digital_transactions
        GROUP BY year ORDER BY year
        """,
        conn,
    )
    conn.close()

    client = MongoClient("mongodb://localhost:27017")
    db     = client["dsm"]
    wireless_ts = get_national_wireless_ts(db)
    client.close()

    wireless_annual = (
        wireless_ts.groupby("year")["total_wireless"]
        .mean().reset_index()
        .rename(columns={"total_wireless": "wireless_mean"})
    )

    def normalize(s):
        return (s - s.min()) / (s.max() - s.min())

    elec["normalized"]            = normalize(elec["energy_gwh"])
    wireless_annual["normalized"] = normalize(wireless_annual["wireless_mean"])
    digital["normalized"]         = normalize(digital["digital_txn_mean"])

    return {"elec": elec, "wireless": wireless_annual, "digital": digital}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.header("Connectivity & the Digital Economy")

st.info(
    "**RQ 3** — Did wireless subscriber growth cause growth in digital transaction volumes? "
    "Does the co-movement of electricity, wireless subscribers, and digital payments provide "
    "corroborating evidence that connectivity expansion enabled the digital economy?\n\n"
    "We test this with a **Granger causality F-test** on monthly growth rates (lags 1–4 months), "
    "and cross-validate using a normalized time-series overlay of three independent indicators."
)

st.divider()

# ---------------------------------------------------------------------------
# Section 2 — Corroboration
# ---------------------------------------------------------------------------

st.subheader("1. Long-run Corroboration of Electricity, Wireless & Digital Transactions")
st.markdown(
    "To check whether the wireless–payments link is real over a longer horizon, we normalize "
    "three independent indicators to a 0–1 scale and overlay them. If all three rise together "
    "and their inflection points align with known policy events, it supports a shared underlying "
    "driver: India's digital infrastructure build-out."
)

corr_data = load_corroboration_data()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=corr_data["elec"]["year"],
    y=corr_data["elec"]["normalized"],
    mode="lines+markers",
    name="Domestic + Commercial Electricity",
    line=dict(color=BLUE),
))
fig.add_trace(go.Scatter(
    x=corr_data["wireless"]["year"],
    y=corr_data["wireless"]["normalized"],
    mode="lines+markers",
    name="Wireless Subscribers",
    line=dict(color=ORANGE),
))
fig.add_trace(go.Scatter(
    x=corr_data["digital"]["year"],
    y=corr_data["digital"]["normalized"],
    mode="lines+markers",
    name="Digital Transactions",
    line=dict(color=GREEN),
))

fig.add_vline(x=2010, line_dash="dash", line_color="gray")
fig.add_annotation(
    x=2010, y=1, yref="paper", text="Digital acceleration (2010)",
    showarrow=False, xanchor="left", font=dict(color="gray", size=11),
)
fig.add_vline(x=2016, line_dash="dash", line_color=RED)
fig.add_annotation(
    x=2016, y=0.88, yref="paper", text="Jio entry (2016)",
    showarrow=False, xanchor="left", font=dict(color=RED, size=11),
)

fig.update_layout(
    xaxis=dict(title="Year", range=[1990, 2023]),
    yaxis=dict(title="Normalized Value (0–1)"),
    legend=dict(font=dict(size=10)),
    margin=dict(l=20, r=20, t=30, b=20),
    height=420,
)
st.plotly_chart(fig, width="stretch")

st.markdown(
    "> **Finding:** All three indicators rise together from 2010 onward, with a visible "
    "acceleration in wireless subscribers and digital transactions post-2016 (Jio entry). "
    "Electricity consumption grows more steadily, confirming it captures broader economic "
    "development rather than the Jio-specific shock. The alignment of all three curves "
    "post-2016 is consistent with Jio's free-data strategy unlocking digital payment adoption "
    "at scale. Together, these corroborate the hypothesis that connectivity expansion and "
    "digital economic activity are tightly linked — even where the monthly Granger test "
    "does not reach significance."
)

st.caption(
    "Electricity: Domestic + Commercial sectors (CEA data, 1990–2023). "
    "Wireless: national monthly subscriber count averaged by year (TRAI/MongoDB). "
    "Digital Transactions: monthly RBI data averaged by year. All series normalized to [0, 1]."
)

# ---------------------------------------------------------------------------
# Section 1 — Granger Causality
# ---------------------------------------------------------------------------

st.subheader("2. Granger Causality — Does Wireless Growth Predict Digital Transaction Growth?")
st.markdown(
    "Granger causality asks: does knowing past values of *wireless subscriber growth rate* improve "
    "our forecast of *digital transaction growth rate*, beyond what past transaction data alone tells us? "
    "We test this at four different lags (1 through 4 months)."
)

granger_df = compute_granger_results()
st.dataframe(granger_df, width="stretch", hide_index=True)

any_significant = (granger_df["Significant"] == "Yes ✓").any()
if any_significant:
    sig_lags = granger_df[granger_df["Significant"] == "Yes ✓"]["Lag (months)"].tolist()
    st.markdown(
        f"> **Finding:** Wireless growth *does* Granger-cause digital transaction growth at "
        f"lag(s) {sig_lags} month(s) (p < 0.05). This means that a rise in wireless subscribers "
        f"in one month is statistically predictive of higher digital payment volumes in the "
        f"following {max(sig_lags)} month(s) — consistent with the hypothesis that connectivity "
        f"expansion enabled the digital payments ecosystem."
    )
else:
    st.markdown(
        "> **Finding:** No significant Granger causality detected at any tested lag (1–4 months). "
        "Wireless subscriber growth may not directly predict digital transaction growth at the "
        "monthly frequency. This could reflect temporal mismatch — infrastructure rollout effects "
        "on payment behaviour may play out over quarters or years rather than months. "
        "The corroboration evidence below, however, shows strong long-run co-movement."
    )

st.divider()

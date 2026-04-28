import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.stattools import grangercausalitytests

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.dashboard.data_loader import load_wireless_ts, load_digital_transactions, SQLITE_PATH

BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN  = "#2ca02c"
RED    = "#d62728"

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

# runs Granger causality test on wireless and digital transaction growth
@st.cache_data(ttl=3600)
def compute_granger_results():
    digital_txn = load_digital_transactions()
    wireless = load_wireless_ts()

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
            "Significant":  "Yes" if p_value < 0.05 else "No",
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

    wireless_ts = load_wireless_ts()
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
with st.expander("What is Granger causality?"):
    st.markdown(
        "**Granger causality** is a statistical test for whether one time series is useful in "
        "forecasting another. It does NOT prove true causation -- it tests *predictive precedence*.\n\n"
        "The test works by comparing two models:\n\n"
        "1. **Restricted model**: predict `txn_growth(t)` using only its own past values "
        "(lags of `txn_growth`)\n"
        "2. **Unrestricted model**: predict `txn_growth(t)` using its own past values PLUS "
        "past values of `wireless_growth`\n\n"
        "If adding `wireless_growth` lags significantly improves the forecast (measured by an "
        "F-test comparing residual sums of squares), we say wireless growth *Granger-causes* "
        "digital transaction growth.\n\n"
        "We test at lags 1 through 4 months. Both series are converted to **percentage-change "
        "growth rates** first (to ensure stationarity, which is required for the test to be valid). "
        "A p-value < 0.05 at any lag means wireless growth has significant predictive power "
        "for transaction growth at that time horizon."
    )

granger_df = compute_granger_results()
st.dataframe(granger_df, width="stretch", hide_index=True)

any_significant = (granger_df["Significant"] == "Yes").any()
if any_significant:
    sig_lags = granger_df[granger_df["Significant"] == "Yes"]["Lag (months)"].tolist()
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

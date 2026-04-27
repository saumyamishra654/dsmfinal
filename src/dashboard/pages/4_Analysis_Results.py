"""
Page 4 — Analysis Results

Structural break detection (Bai-Perron), Granger causality testing,
and electricity corroboration with wireless and digital transactions.
"""

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

from src.analysis.obj1_wireless_growth import (
    detect_structural_breaks,
    chow_test,
    compute_cagr,
)
from src.analysis.obj2_teledensity_ger import (
    get_yearly_correlation,
    get_regression_results,
)
from src.dashboard.data_loader import (
    load_wireless_ts,
    load_digital_transactions,
    SQLITE_PATH,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def compute_structural_break_results() -> dict:
    """Run structural break detection, Chow tests, and CAGR computation."""
    ts = load_wireless_ts()
    series = ts["total_wireless"].values

    break_indices = detect_structural_breaks(series, n_bkps=2)
    break_dates = [ts["date"].iloc[min(idx, len(ts) - 1)] for idx in break_indices]

    chow_rows = []
    for i, (idx, dt) in enumerate(zip(break_indices, break_dates)):
        f_stat, p_value = chow_test(series, idx)
        chow_rows.append({
            "Break": i + 1,
            "Date": dt.strftime("%b %Y"),
            "Index": idx,
            "F-statistic": round(f_stat, 2),
            "p-value": f"{p_value:.2e}",
            "Significant": "Yes" if p_value < 0.05 else "No",
        })

    first_break = break_dates[0]
    second_break = break_dates[1]
    periods = [
        ("Pre-break 1", ts[ts["date"] < first_break]),
        ("Break 1 to Break 2", ts[(ts["date"] >= first_break) & (ts["date"] < second_break)]),
        ("Post-break 2", ts[ts["date"] >= second_break]),
    ]

    cagr_rows = []
    for label, p in periods:
        if len(p) < 2:
            continue
        years = (p["date"].iloc[-1] - p["date"].iloc[0]).days / 365.25
        cagr = compute_cagr(
            p["total_wireless"].iloc[0],
            p["total_wireless"].iloc[-1],
            years,
        )
        cagr_rows.append({
            "Period": label,
            "Start": p["date"].iloc[0].strftime("%b %Y"),
            "End": p["date"].iloc[-1].strftime("%b %Y"),
            "CAGR (%)": round(cagr * 100, 2),
        })

    return {
        "chow_rows": chow_rows,
        "cagr_rows": cagr_rows,
        "break_dates": [d.isoformat() for d in break_dates],
    }


@st.cache_data(ttl=3600)
def compute_granger_results() -> pd.DataFrame:
    """Granger causality: wireless growth -> digital transaction growth."""
    digital_txn = load_digital_transactions()
    wireless = load_wireless_ts()

    merged = pd.merge(
        digital_txn[["year", "month", "digital_txn_crores"]],
        wireless[["year", "month", "total_wireless"]],
        on=["year", "month"],
        how="inner",
    )
    merged = merged.sort_values(["year", "month"]).reset_index(drop=True)

    merged["wireless_growth"] = merged["total_wireless"].pct_change()
    merged["txn_growth"] = merged["digital_txn_crores"].pct_change()
    merged = merged.dropna(subset=["wireless_growth", "txn_growth"])

    data = merged[["txn_growth", "wireless_growth"]].values
    maxlag = 4
    results = grangercausalitytests(data, maxlag=maxlag, verbose=False)

    rows = []
    for lag in range(1, maxlag + 1):
        f_stat = results[lag][0]["ssr_ftest"][0]
        p_value = results[lag][0]["ssr_ftest"][1]
        rows.append({
            "Lag (months)": lag,
            "F-statistic": round(f_stat, 4),
            "p-value": round(p_value, 4),
            "Significant": "Yes" if p_value < 0.05 else "No",
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def load_corroboration_data() -> dict:
    """Load and normalize electricity, wireless, and digital transaction data."""
    conn = sqlite3.connect(str(SQLITE_PATH))

    elec = pd.read_sql_query(
        """
        SELECT year, SUM(energy_gwh) AS energy_gwh
        FROM electricity_consumption
        WHERE sector IN ('Domestic', 'Commercial')
        GROUP BY year
        ORDER BY year
        """,
        conn,
    )

    digital = pd.read_sql_query(
        """
        SELECT year, AVG(digital_txn_crores) AS digital_txn_mean
        FROM digital_transactions
        GROUP BY year
        ORDER BY year
        """,
        conn,
    )
    conn.close()

    wireless_ts = load_wireless_ts()
    wireless_annual = (
        wireless_ts.groupby("year")["total_wireless"]
        .mean()
        .reset_index()
        .rename(columns={"total_wireless": "wireless_mean"})
    )

    def normalize(s: pd.Series) -> pd.Series:
        return (s - s.min()) / (s.max() - s.min())

    elec["normalized"] = normalize(elec["energy_gwh"])
    wireless_annual["normalized"] = normalize(wireless_annual["wireless_mean"])
    digital["normalized"] = normalize(digital["digital_txn_mean"])

    return {
        "elec": elec,
        "wireless": wireless_annual,
        "digital": digital,
    }


@st.cache_data(ttl=3600)
def load_obj2_correlation() -> pd.DataFrame:
    return get_yearly_correlation()


@st.cache_data(ttl=3600)
def load_obj2_regression() -> list[dict]:
    return get_regression_results()


# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.header("Analysis Results")

# ── Section 1: Structural Break Detection ─────────────────────────────────

st.subheader("Structural Break Detection (Bai-Perron)")

break_results = compute_structural_break_results()

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Chow Test Results**")
    st.dataframe(
        pd.DataFrame(break_results["chow_rows"]),
        width="stretch",
        hide_index=True,
    )

with col_right:
    st.markdown("**CAGR by Period**")
    st.dataframe(
        pd.DataFrame(break_results["cagr_rows"]),
        width="stretch",
        hide_index=True,
    )

# ── Section 2: Granger Causality ──────────────────────────────────────────

st.subheader("Granger Causality: Wireless Growth \u2192 Digital Transaction Growth")

granger_df = compute_granger_results()

st.dataframe(granger_df, width="stretch", hide_index=True)

if (granger_df["Significant"] == "No").all():
    st.warning(
        "No significant Granger causality detected at any tested lag. "
        "Wireless subscriber growth may not directly Granger-cause digital "
        "transaction growth at the monthly frequency."
    )

# ── Section 3: Electricity Corroboration ──────────────────────────────────

st.subheader("Corroboration: Electricity \u00d7 Wireless \u00d7 Digital Transactions")

corr_data = load_corroboration_data()

fig = go.Figure()

# Domestic + Commercial electricity
fig.add_trace(go.Scatter(
    x=corr_data["elec"]["year"],
    y=corr_data["elec"]["normalized"],
    mode="lines+markers",
    name="Domestic+Commercial Electricity",
    line=dict(color=BLUE),
))

# Wireless subscribers
fig.add_trace(go.Scatter(
    x=corr_data["wireless"]["year"],
    y=corr_data["wireless"]["normalized"],
    mode="lines+markers",
    name="Wireless Subscribers",
    line=dict(color=ORANGE),
))

# Digital transactions
fig.add_trace(go.Scatter(
    x=corr_data["digital"]["year"],
    y=corr_data["digital"]["normalized"],
    mode="lines+markers",
    name="Digital Transactions",
    line=dict(color=GREEN),
))

# Vertical annotation lines (annotation_text omitted — causes Plotly bug on integer axes)
fig.add_vline(x=2010, line_dash="dash", line_color="gray")
fig.add_annotation(x=2010, y=1, yref="paper", text="Digital acceleration",
                   showarrow=False, xanchor="left", font=dict(color="gray"))
fig.add_vline(x=2016, line_dash="dash", line_color=RED)
fig.add_annotation(x=2016, y=0.9, yref="paper", text="Jio entry",
                   showarrow=False, xanchor="left", font=dict(color=RED))

fig.update_layout(
    xaxis=dict(title="Year", range=[1990, 2023]),
    yaxis=dict(title="Normalized Value (0\u20131)"),
    legend=dict(font=dict(size=10)),
    margin=dict(l=20, r=20, t=30, b=20),
)

st.plotly_chart(fig, width="stretch")

# ── Section 4: Tele-density vs GER (Objective 2) ──────────────────────────

st.divider()
st.subheader("Tele-density vs GER — Panel Regression (Objective 2)")

corr_df  = load_obj2_correlation()
reg_results = load_obj2_regression()

col_left4, col_right4 = st.columns(2)

with col_left4:
    st.markdown("**Annual Cross-State Correlation: Tele-density vs Total GER**")
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(
        x=corr_df["year"], y=corr_df["pearson_r"],
        mode="lines+markers", line=dict(color=BLUE), name="Pearson r",
    ))
    fig_corr.add_vline(x=2016, line_dash="dash", line_color=RED)
    fig_corr.add_annotation(x=2016, y=1, yref="paper", text="Jio (2016)",
                            showarrow=False, xanchor="left", font=dict(color=RED))
    fig_corr.update_layout(
        xaxis_title="Year", yaxis_title="Pearson r",
        margin=dict(t=30, b=20), height=320,
    )
    st.plotly_chart(fig_corr, width="stretch")

with col_right4:
    st.markdown("**Lagged Panel Regression Coefficients (β on tele_density at t-1)**")
    label_map = {"ger_total": "Total GER", "ger_female": "Female GER", "ger_scst": "SC/ST GER"}
    reg_df = pd.DataFrame([{
        "GER Group":   label_map[r["dep_var"]],
        "β":           r["coef"],
        "Std Error":   r["std_err"],
        "t-stat":      r["t_stat"],
        "p-value":     r["p_value"],
        "Significant": "Yes" if r["p_value"] < 0.05 else "No",
        "R²(within)":  r["r2_within"],
        "N":           r["n_obs"],
    } for r in reg_results])
    st.dataframe(reg_df, width="stretch", hide_index=True)
    st.caption(
        "Two-way fixed effects (state + year). Clustered standard errors. "
        "None of the coefficients reach p < 0.05 — direction is positive for "
        "Total and Female GER but the panel (18 states × 8 years) is too short "
        "to detect small effects after absorbing fixed effects."
    )

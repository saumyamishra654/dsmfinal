import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.analysis.obj2_teledensity_ger import (
    get_panel,
    get_yearly_correlation,
    get_regression_results,
)

BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
RED    = "#d62728"

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

# loads the tele-density vs GER panel
@st.cache_data(ttl=3600)
def load_panel():
    return get_panel()

# loads the yearly cross-state Pearson correlation
@st.cache_data(ttl=3600)
def load_correlation():
    return get_yearly_correlation()

# runs the three lagged panel regressions
@st.cache_data(ttl=3600)
def load_regression():
    return get_regression_results()

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.header("Connectivity & Education")

st.info(
    "**RQ 2** — Does increased telecom access (tele-density) translate into higher Gross Enrolment "
    "Ratio (GER) at the state level? Is the effect different for women and SC/ST groups?\n\n"
    "We test this using a **lagged two-way fixed-effects panel regression** "
    "(tele-density at t−1 predicting GER at t, 17 states × 2013–2021)."
)

panel = load_panel()

st.divider()

# ---------------------------------------------------------------------------
# Section 1 — Scatter
# ---------------------------------------------------------------------------

st.subheader("1. Tele-density vs GER — Cross-State Scatter")
st.markdown(
    "Each point is one state in one year. The upward slope shows that states with higher "
    "tele-density tend to have higher education enrolment — but this is a **levels relationship**, "
    "partly driven by between-state wealth differences rather than connectivity alone."
)

valid = panel.dropna(subset=["tele_density", "ger_total"])
m, b  = np.polyfit(valid["tele_density"], valid["ger_total"], 1)
x_line = np.linspace(valid["tele_density"].min(), valid["tele_density"].max(), 100)

fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=valid["tele_density"],
    y=valid["ger_total"],
    mode="markers",
    text=valid["state"] + " (" + valid["year"].astype(str) + ")",
    hovertemplate="%{text}<br>Tele-density: %{x:.1f}<br>GER: %{y:.1f}<extra></extra>",
    marker=dict(color=BLUE, opacity=0.55, size=8, line=dict(width=0.5, color="white")),
    name="State-Year observation",
))
fig_scatter.add_trace(go.Scatter(
    x=x_line,
    y=m * x_line + b,
    mode="lines",
    line=dict(color=RED, width=2, dash="dash"),
    name=f"OLS fit (slope = {m:.3f})",
))
fig_scatter.update_layout(
    xaxis_title="Tele-density (phones per 100 people)",
    yaxis_title="Gross Enrolment Ratio — Total",
    margin=dict(l=20, r=20, t=30, b=20),
    height=420,
    legend=dict(font=dict(size=11)),
)
st.plotly_chart(fig_scatter, width="stretch")

st.markdown(
    "> **Finding:** The cross-state OLS slope is positive (~0.08–0.12), confirming "
    "that higher-connectivity states tend to have higher GER. However, this relationship "
    "is partly spurious — both tele-density and GER are correlated with state income level. "
    "The panel regression below controls for this by absorbing state fixed effects."
)

st.divider()


# ---------------------------------------------------------------------------
# Section 3 — Panel regression
# ---------------------------------------------------------------------------

st.subheader("2. Lagged Panel Regression — Does Tele-density Predict GER?")
st.markdown(
    "To test causal direction, we use **tele-density at year t−1** to predict GER at year t."
)

reg_results = load_regression()
label_map   = {"ger_total": "Total GER", "ger_female": "Female GER", "ger_scst": "SC/ST GER"}

reg_df = pd.DataFrame([{
    "GER Group":    label_map[r["dep_var"]],
    "β (tele-density t−1)": r["coef"],
    "Std Error":    r["std_err"],
    "t-stat":       r["t_stat"],
    "p-value":      r["p_value"],
    "Significant":  "Yes" if r["p_value"] < 0.05 else "No",
    "R² (within)":  r["r2_within"],
    "N obs":        r["n_obs"],
} for r in reg_results])

st.dataframe(reg_df, width="stretch", hide_index=True)

# Coefficient bar chart
fig_coef = go.Figure()
for r in reg_results:
    ci = 1.96 * r["std_err"]
    color = BLUE if r["p_value"] < 0.05 else "#b0b0b0"
    fig_coef.add_trace(go.Bar(
        x=[label_map[r["dep_var"]]],
        y=[r["coef"]],
        error_y=dict(type="data", array=[ci], visible=True, color="black", thickness=1.5),
        marker_color=color,
        name=label_map[r["dep_var"]],
        showlegend=False,
        hovertemplate=f"β = {r['coef']}<br>95% CI ± {ci:.4f}<br>p = {r['p_value']}<extra></extra>",
    ))
fig_coef.add_hline(y=0, line_color="black", line_width=1)
fig_coef.update_layout(
    xaxis_title="GER Group",
    yaxis_title="β on tele-density (t−1)",
    title="Regression Coefficients with 95% CI  (grey = not significant at p < 0.05)",
    margin=dict(l=20, r=20, t=50, b=20),
    height=340,
)
st.plotly_chart(fig_coef, width="stretch")

st.markdown(
    "> **Finding:** All three coefficients are positive for Total and Female GER — the direction "
    "is consistent with the hypothesis that connectivity growth precedes education gains. "
    "However, **none reach statistical significance** (all p > 0.10). This is likely a "
    "**statistical power problem**: with only 18 states and 8 time periods, absorbing two sets "
    "of fixed effects leaves too little variation to detect a small, real effect. "
    "We treat this as *suggestive but inconclusive* evidence."
)

st.caption(
    "Model: Two-way fixed effects (state + year). Predictor: annual mean tele-density at t−1. "
    "Standard errors clustered by state. SC/ST GER = average of Scheduled Caste and Scheduled Tribe GER "
    "(Total gender, all categories). Panel: 17 states × 2014–2021 (first year dropped for lag)."
)

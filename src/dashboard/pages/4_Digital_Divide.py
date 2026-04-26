import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.analysis.obj4_clustering import (
    get_cluster_data,
    get_louvain_graph,
    get_gap_analysis,
)

COMMUNITY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

# loads the full feature matrix with cluster labels
@st.cache_data(ttl=3600)
def load_cluster_data():
    return get_cluster_data()


# loads the gap analysis table for laggard states
@st.cache_data(ttl=3600)
def load_gap_analysis():
    features = get_cluster_data()
    return get_gap_analysis(features)


# builds a Plotly network graph of the Louvain communities
def build_louvain_plotly(features):
    G, partition, pos = get_louvain_graph(features)

    weights   = [d["weight"] for _, _, d in G.edges(data=True)]
    threshold = float(np.median(weights))

    edge_traces = []
    for u, v, d in G.edges(data=True):
        if d["weight"] < threshold:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=d["weight"] * 3, color="lightgrey"),
            hoverinfo="none",
            showlegend=False,
        ))

    communities = sorted(set(partition.values()))
    node_traces = []
    for comm in communities:
        members = [s for s in G.nodes() if partition[s] == comm]
        x_vals  = [pos[s][0] for s in members]
        y_vals  = [pos[s][1] for s in members]
        node_traces.append(go.Scatter(
            x=x_vals, y=y_vals,
            mode="markers+text",
            text=members,
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(
                size=18,
                color=COMMUNITY_COLORS[comm % len(COMMUNITY_COLORS)],
                line=dict(width=1, color="white"),
            ),
            name=f"Community {comm}",
            hovertemplate="%{text}<extra></extra>",
        ))

    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(
        title="Louvain Community Detection — State Similarity Graph",
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20),
        height=520,
        legend=dict(title="Community", font=dict(size=11)),
    )
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.header("The Digital Divide")

st.info(
    "**RQ 4** — Which Indian states are digitally excluded? How are states clustered by "
    "connectivity profile, and how many years would it take laggard states to reach the "
    "leader group at their current growth rates?\n\n"
    "We construct a **state-similarity graph** (edge weight = cosine similarity of each state's "
    "tele-density and GER features) and run **Louvain community detection** to identify natural "
    "groupings. A forward-looking gap analysis then quantifies how far behind the laggard "
    "community is."
)

features = load_cluster_data()

st.divider()

# ---------------------------------------------------------------------------
# Section 1 — Louvain graph
# ---------------------------------------------------------------------------

st.subheader("1. State Similarity Graph (Louvain Communities)")
st.markdown(
    "Each **node** is an Indian state. Each **edge** represents cosine similarity between two "
    "states' feature vectors (mean tele-density, tele-density growth slope, mean GER, GER growth "
    "slope). Only edges above the median similarity are shown to reduce clutter. "
    "**Node colour** indicates Louvain community membership — states of the same colour are "
    "structurally similar in their connectivity and education profiles."
)

fig_network = build_louvain_plotly(features)
st.plotly_chart(fig_network, width="stretch")

st.markdown(
    "> **Finding:** Louvain detects **3 communities**, each with a distinct connectivity profile:\n"
    "> - **Community 0 (blue):** High tele-density southern and metro states (Kerala, Tamil Nadu, "
    "Karnataka, Maharashtra, Delhi, AP, HP) — the digital leaders.\n"
    "> - **Community 1 (orange):** Low tele-density northern and eastern states (Bihar, Assam, UP, "
    "West Bengal, MP, Odisha, Rajasthan, Gujarat) — the digital laggards.\n"
    "> - **Community 2 (green):** Mid-tier northern states (Haryana, Punjab, J&K) — between the "
    "two extremes in both tele-density and GER.\n\n"
    "> This three-way split is more nuanced than K-means (which found only 2 clusters), capturing "
    "the Punjab/Haryana/J&K mid-tier that K-means lumped into the mainstream cluster."
)

st.divider()

# ---------------------------------------------------------------------------
# Section 2 — Community membership + profiles
# ---------------------------------------------------------------------------

st.subheader("2. Community Membership & Profiles")

communities = sorted(features["louvain_community"].unique())
cols = st.columns(len(communities))
for col, comm in zip(cols, communities):
    members = features[features["louvain_community"] == comm]["state"].tolist()
    color   = COMMUNITY_COLORS[comm % len(COMMUNITY_COLORS)]
    col.markdown(
        f"<div style='border-left: 4px solid {color}; padding-left: 8px;'>"
        f"<b>Community {comm}</b><br>" + "<br>".join(members) + "</div>",
        unsafe_allow_html=True,
    )

st.markdown("#### Mean Feature Values by Community")
st.caption(
    "Mean tele-density = phones per 100 people (averaged over 2013–2021). "
    "Mean GER = Gross Enrolment Ratio, Total, All Categories (averaged over available years)."
)

profile = (
    features.groupby("louvain_community")[["mean_tele_density", "mean_ger"]]
    .mean()
    .round(1)
    .reset_index()
    .rename(columns={
        "louvain_community": "Community",
        "mean_tele_density":  "Mean Tele-density",
        "mean_ger":           "Mean GER",
    })
)
st.dataframe(profile, width="stretch", hide_index=True)

st.markdown(
    "> **Finding:** The leader community (Community 0) has a mean tele-density of ~129 — "
    "nearly **1.8× the laggard community's ~72**. The GER gap is similarly stark: ~36 vs ~20. "
    "This confirms that digital and educational exclusion are spatially co-located: the same "
    "states that lag on connectivity also lag on education enrolment."
)

st.divider()

# ---------------------------------------------------------------------------
# Section 3 — Gap analysis
# ---------------------------------------------------------------------------

st.subheader("3. Gap Analysis: How Far Behind Are the Laggard States?")
st.markdown(
    "For each state in the lowest tele-density community, we compute: "
    "**(leader mean tele-density − state mean) ÷ state annual tele-density growth rate**. "
    "This gives the number of years — at the state's current growth pace — needed to reach "
    "the leader community's mean. States with a *negative* growth rate are excluded as they "
    "are actively diverging from the leader."
)

gap_df = load_gap_analysis()
gap_df["years_to_close_gap"] = pd.to_numeric(gap_df["years_to_close_gap"], errors="coerce")
gap_df = (
    gap_df
    .dropna(subset=["years_to_close_gap"])
    .sort_values("years_to_close_gap", ascending=False)
    .head(5)
)

gap_df = gap_df.rename(columns={
    "state":              "State",
    "tele_density_mean":  "Mean Tele-density",
    "annual_growth":      "Annual Growth (units/yr)",
    "gap_to_leader":      "Gap to Leader",
    "years_to_close_gap": "Years to Close Gap",
})

st.dataframe(gap_df, width="stretch", hide_index=True)

st.markdown(
    "> **Finding:** Bihar and Rajasthan are **structural laggards**: not merely behind, but "
    "growing so slowly that at current rates they would need over a century to reach the "
    "leader community."
    "Without targeted policy intervention, such as subsidised infrastructure, mandatory rollout "
    "obligations, or direct device/data subsidies, the digital divide in these states will "
    "not close within any planning horizon."
)

st.caption(
    "Leader community = the Louvain community with the highest mean tele-density. "
    "Annual growth = OLS slope of tele-density against year index, 2013–2021."
)

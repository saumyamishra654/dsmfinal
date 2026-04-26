"""
Page 3 — Digital Divide (Placeholder).

Placeholder page for clustering results (K-means + Louvain community
detection). Shows state tele-density rankings until Objective 4 is complete.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "db" / "sqlite" / "dsm.db"

# ── Data loader ────────────────────────────────────────────────────────────


@st.cache_data(ttl=3600)
def load_tele_density() -> pd.DataFrame:
    """Fetch state tele-density for the latest available month."""
    query = """
        SELECT s.state_name, td.tele_density, td.year
        FROM tele_density td
        JOIN states s ON td.state_id = s.state_id
        WHERE td.year = (SELECT MAX(year) FROM tele_density)
          AND td.month = (
              SELECT MAX(month) FROM tele_density
              WHERE year = (SELECT MAX(year) FROM tele_density)
          )
        ORDER BY td.tele_density DESC
    """
    with sqlite3.connect(str(DB_PATH)) as conn:
        return pd.read_sql_query(query, conn)


# ── Page content ───────────────────────────────────────────────────────────

st.header("📉 Digital Divide Analysis")

st.info(
    "🚧 Clustering analysis pending — will show choropleth map of digital "
    "divide clusters (K-means + Louvain community detection) once Objective 4 "
    "is complete. For now, see state rankings below."
)

df = load_tele_density()

if df.empty:
    st.warning("No tele-density data found in the database.")
    st.stop()

col1, col2 = st.columns([2, 1])

# ── Left column: horizontal bar chart ─────────────────────────────────────

with col1:
    fig = px.bar(
        df,
        x="tele_density",
        y="state_name",
        orientation="h",
        color="tele_density",
        color_continuous_scale="RdYlGn",
        labels={
            "tele_density": "Tele-density",
            "state_name": "State",
        },
        title=f"States Ranked by Tele-density ({int(df['year'].iloc[0])})",
    )
    fig.update_layout(yaxis={"autorange": "reversed"})
    st.plotly_chart(fig, use_container_width=True)

# ── Right column: summary metrics ─────────────────────────────────────────

with col2:
    highest_state = df.iloc[0]["state_name"]
    highest_val = df.iloc[0]["tele_density"]

    lowest_state = df.iloc[-1]["state_name"]
    lowest_val = df.iloc[-1]["tele_density"]

    median_val = np.median(df["tele_density"])
    above_100 = int((df["tele_density"] > 100).sum())

    st.metric("Highest", f"{highest_state}", f"{highest_val:.1f}")
    st.metric("Lowest", f"{lowest_state}", f"{lowest_val:.1f}")
    st.metric("Median Tele-density", f"{median_val:.1f}")
    st.metric("States above 100", f"{above_100}")

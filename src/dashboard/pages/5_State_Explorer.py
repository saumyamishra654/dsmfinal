import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Database path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "db" / "sqlite" / "dsm.db"


def _get_connection():
    return sqlite3.connect(str(DB_PATH))


# returns list of distinct state names that have tele-density data
@st.cache_data
def get_state_names():
    query = """
        SELECT DISTINCT s.state_name
        FROM states s
        JOIN tele_density td ON s.state_id = td.state_id
        ORDER BY s.state_name
    """
    with _get_connection() as conn:
        df = pd.read_sql_query(query, conn)
    return df["state_name"].tolist()


# loads monthly tele-density for one state
@st.cache_data
def get_tele_density(state):
    query = """
        SELECT td.year, td.month, td.date, td.tele_density
        FROM tele_density td
        JOIN states s ON td.state_id = s.state_id
        WHERE s.state_name = ?
        ORDER BY td.date
    """
    with _get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=(state,))
    df["date"] = pd.to_datetime(df["date"])
    return df


# loads monthly wired and wireless subscriber counts for one state
@st.cache_data
def get_wired_wireless(state):
    query = """
        SELECT ww.date, ww.wireline_millions, ww.wireless_millions
        FROM wired_wireless ww
        JOIN states s ON ww.state_id = s.state_id
        WHERE s.state_name = ?
        ORDER BY ww.date
    """
    with _get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=(state,))
    df["date"] = pd.to_datetime(df["date"])
    return df


# loads GER for one state, gender, and category
@st.cache_data
def get_ger(state, gender, category):
    query = """
        SELECT eg.year, eg.ger
        FROM education_ger eg
        JOIN states s ON eg.state_id = s.state_id
        WHERE s.state_name = ?
          AND eg.gender = ?
          AND eg.category = ?
        ORDER BY eg.year
    """
    with _get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=(state, gender, category))
    return df


# line chart of tele-density over time for one or two states
def _chart_tele_density(states):
    fig = go.Figure()
    for s in states:
        df = get_tele_density(s)
        fig.add_trace(go.Scatter(x=df["date"], y=df["tele_density"],
                                 mode="lines", name=s))
    fig.update_layout(
        title="Tele-density Over Time",
        xaxis_title="Date",
        yaxis_title="Tele-density",
        height=350,
        margin=dict(t=30, b=30),
    )
    return fig


# wireless vs wireline subscriber chart
def _chart_wired_wireless(state):
    df = get_wired_wireless(state)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["wireless_millions"],
        mode="lines", name="Wireless",
        line=dict(color="#1f77b4"),
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["wireline_millions"],
        mode="lines", name="Wireline",
        line=dict(color="#ff7f0e"),
    ))
    fig.update_layout(
        title=f"Wired vs Wireless — {state}",
        xaxis_title="Date",
        yaxis_title="Subscribers (millions)",
        height=350,
        margin=dict(t=30, b=30),
    )
    return fig


# GER lines+markers for selected states, gender, and category
def _chart_ger(states, gender, category):
    fig = go.Figure()
    for s in states:
        df = get_ger(s, gender, category)
        fig.add_trace(go.Scatter(
            x=df["year"], y=df["ger"],
            mode="lines+markers", name=s,
        ))
    fig.update_layout(
        title=f"Gross Enrolment Ratio — {gender}, {category}",
        xaxis_title="Year",
        yaxis_title="GER",
        height=350,
        margin=dict(t=30, b=30),
    )
    return fig


st.header("State Explorer")
st.markdown(
    "Use this page to explore any individual state's telecom and education trajectory. "
    "Select one state to explore its trends, or toggle **Compare two states** to place them "
    "side by side — useful for illustrating the gap between a leader state (e.g. Kerala) "
    "and a laggard (e.g. Bihar or Rajasthan) identified in the Digital Divide analysis."
)
st.caption(
    "Coverage: 17 major states where TRAI telecom circle names map cleanly to state names. "
    "GER data: 2012–2021. Tele-density and subscriber data: 2013–2023."
)

st.divider()

state_names = get_state_names()

if not state_names:
    st.warning("No state data found in the database.")
    st.stop()

# --- Controls ---
compare = st.toggle("Compare two states")

if compare:
    col_a, col_b = st.columns(2)
    with col_a:
        state_1 = st.selectbox("State 1", state_names, index=0)
    with col_b:
        state_2 = st.selectbox("State 2", state_names,
                                index=min(1, len(state_names) - 1))
    selected_states = [state_1, state_2]
else:
    state_1 = st.selectbox("State", state_names, index=0)
    selected_states = [state_1]

gender = st.radio("GER Gender", ["Total", "Male", "Female"], horizontal=True)
category = st.radio(
    "GER Category",
    ["All Categories", "Scheduled Caste", "Scheduled Tribe"],
    horizontal=True,
)

# --- Chart 1: Tele-density ---
st.subheader("Tele-density")
st.plotly_chart(_chart_tele_density(selected_states), width="stretch")

# --- Chart 2: Wired vs Wireless ---
st.subheader("Wired vs Wireless Subscribers")
if compare:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_chart_wired_wireless(selected_states[0]),
                        width="stretch")
    with col2:
        st.plotly_chart(_chart_wired_wireless(selected_states[1]),
                        width="stretch")
else:
    st.plotly_chart(_chart_wired_wireless(selected_states[0]),
                    width="stretch")

# --- Chart 3: GER ---
st.subheader("Gross Enrolment Ratio (GER)")
st.plotly_chart(
    _chart_ger(selected_states, gender, category),
    width="stretch",
)

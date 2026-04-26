"""
LLM-powered sidebar chat for the Digital India Dashboard.

Lets the user ask natural-language questions about the underlying
SQLite and MongoDB data.  Claude generates Python code that is
executed locally; the result (DataFrame, Plotly figure, or plain
text) is rendered in the main area.
"""

import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pymongo
import sqlite3
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = str(Path(__file__).resolve().parents[2] / "db" / "sqlite" / "dsm.db")
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "dsm"

SYSTEM_PROMPT = """\
You are a data-analysis assistant for the Digital India Dashboard.
The user will ask questions about Indian telecom, digital transactions,
education, and electricity data.

Return ONLY a Python code block (```python ... ```) that assigns
its final output to a variable called `result`.

`result` can be:
- a pandas DataFrame
- a Plotly Figure (px or go)
- a plain string / number

Available namespace (already imported for you):
  pd, np, sqlite3, pymongo, px, go, db_path, mongo_uri, mongo_db

----- SQLite schema (file path: db_path) -----
- states(state_id INTEGER PRIMARY KEY, state_name TEXT UNIQUE)
- tele_density(id INTEGER PRIMARY KEY, state_id INTEGER FK, year INTEGER, month INTEGER, date DATE, tele_density REAL)
- wired_wireless(id INTEGER PRIMARY KEY, state_id INTEGER FK, year INTEGER, month INTEGER, date DATE, wireline_millions REAL, wireless_millions REAL, pct_share REAL)
- education_ger(id INTEGER PRIMARY KEY, state_id INTEGER FK, year INTEGER, gender TEXT, category TEXT, ger REAL)
- digital_transactions(id INTEGER PRIMARY KEY, year INTEGER, month INTEGER, date DATE, digital_txn_crores REAL, bhim_txn_crores REAL, debit_card_crores REAL)
- electricity_consumption(id INTEGER PRIMARY KEY, year INTEGER, sector TEXT, additional_info TEXT, energy_gwh REAL, pct_consumption REAL, pct_growth REAL)

----- MongoDB (uri: mongo_uri, db: mongo_db) -----
Collection: telecom_subscriptions (~58K documents)
Fields: state, telecom_circle, provider, year, month, wireless_subscribers, wireline_subscribers, vlr_proportion
Providers: Bharti Airtel, Reliance Jio, Vodafone, BSNL, Idea Cellular, etc.
Years: 2008-2021

Rules:
1. Always assign the final answer to `result`.
2. For SQLite queries, use sqlite3.connect(db_path).
3. For MongoDB queries, use pymongo.MongoClient(mongo_uri)[mongo_db].
4. Close connections when done.
5. Do NOT call st.* or print(); just assign to `result`.
"""

QUICK_QUERIES = [
    "Top 5 states by tele-density in 2021",
    "Compare wireless growth: Bihar vs Maharashtra",
    "Monthly UPI share trend",
    "Which provider grew fastest after 2016?",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_code(text: str) -> str:
    """Pull Python code from markdown fences; fall back to raw text."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _run_code(code: str) -> tuple:
    """Run generated code in a namespace and return (result, error).

    This intentionally uses exec() — the user explicitly requested
    dynamic code execution for a locally-hosted one-time demo.
    """
    namespace = {
        "pd": pd,
        "np": np,
        "sqlite3": sqlite3,
        "pymongo": pymongo,
        "px": px,
        "go": go,
        "db_path": DB_PATH,
        "mongo_uri": MONGO_URI,
        "mongo_db": MONGO_DB,
    }
    try:
        # Dynamic execution of LLM-generated code (local demo only)
        run = exec  # noqa: S102
        run(code, namespace)
        return namespace.get("result"), None
    except Exception:
        return None, traceback.format_exc()


def _ask_llm(question: str, api_key: str) -> str:
    """Send *question* to Claude and return the raw response text."""
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=api_key,
        max_tokens=2048,
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def render_sidebar_chat() -> None:
    """Render the LLM chat interface (sidebar + main area components)."""

    # ---- Initialise session state ----
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    # ---- Sidebar controls ----
    with st.sidebar:
        st.markdown("### Ask the Data")
        api_key = st.text_input(
            "Anthropic API key",
            type="password",
            key="llm_api_key",
        )

        if not api_key:
            st.info("Enter your Anthropic API key to enable the chat assistant.")
        else:
            st.markdown("**Quick queries**")
            cols = st.columns(2)
            for idx, q in enumerate(QUICK_QUERIES):
                with cols[idx % 2]:
                    if st.button(q, key=f"quick_{idx}", use_container_width=True):
                        st.session_state.pending_query = q

    # ---- Chat input (must be in main area per Streamlit requirement) ----
    user_input = st.chat_input("Ask a question about the data ...")

    # Determine the active query (explicit input takes precedence)
    query = user_input or st.session_state.pop("pending_query", None)

    # ---- Process query ----
    if query and api_key:
        with st.spinner("Thinking ..."):
            raw_response = _ask_llm(query, api_key)
            code = _extract_code(raw_response)
            result, error = _run_code(code)

        st.session_state.chat_history.append(
            {
                "question": query,
                "code": code,
                "result": result,
                "error": error,
            }
        )

    # ---- Display chat history in main area ----
    for entry in st.session_state.chat_history:
        st.markdown(f"**Q:** {entry['question']}")

        if entry["error"]:
            st.error(entry["error"])
        else:
            result = entry["result"]
            if isinstance(result, pd.DataFrame):
                st.dataframe(result)
            elif hasattr(result, "to_plotly_json"):
                st.plotly_chart(result, use_container_width=True)
            else:
                st.write(result)

        with st.expander("Show generated code"):
            st.code(entry["code"], language="python")

        st.divider()

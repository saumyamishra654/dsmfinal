"""
Digital India Dashboard — Streamlit entry point.

Run:
    streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Digital India Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Page navigation ---
pages = [
    st.Page("pages/1_National_Overview.py", title="National Overview", icon="📊"),
    st.Page("pages/2_State_Explorer.py", title="State Explorer", icon="🗺️"),
    st.Page("pages/3_Digital_Divide.py", title="Digital Divide", icon="📉"),
    st.Page("pages/4_Analysis_Results.py", title="Analysis Results", icon="🔬"),
]
nav = st.navigation(pages)

# --- LLM sidebar chat (visible on every page) ---
from llm_chat import render_sidebar_chat

render_sidebar_chat()

# --- Run selected page ---
nav.run()

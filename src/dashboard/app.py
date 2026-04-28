import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st

WEB_MODE = os.getenv("WEB_MODE", "false").lower() == "true"

st.set_page_config(
    page_title="Digital India Dashboard",
    page_icon="bar_chart",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("pages/0_Overview.py",              title="Study Overview"),
    st.Page("pages/1_National_Overview.py",      title="Telecom Transformation"),
    st.Page("pages/2_Connectivity_Education.py", title="Connectivity & Education"),
    st.Page("pages/3_Digital_Economy.py",        title="Digital Payments"),
    st.Page("pages/4_Digital_Divide.py",         title="The Digital Divide"),
    st.Page("pages/5_State_Explorer.py",         title="State Explorer"),
    st.Page("pages/6_Dataset_Explorer.py",      title="Dataset Explorer"),
]
nav = st.navigation(pages)

if not WEB_MODE:
    from llm_chat import render_sidebar_chat
    render_sidebar_chat()

nav.run()

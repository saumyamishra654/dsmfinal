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

pages = [
    st.Page("pages/0_Overview.py",              title="Study Overview",           icon="🎯"),
    st.Page("pages/1_National_Overview.py",      title="Telecom Transformation",  icon="📡"),
    st.Page("pages/2_Connectivity_Education.py", title="Connectivity & Education", icon="📚"),
    st.Page("pages/3_Digital_Economy.py",        title="Digital Payments",  icon="💳"),
    st.Page("pages/4_Digital_Divide.py",         title="The Digital Divide",       icon="🗺️"),
    st.Page("pages/5_State_Explorer.py",         title="State Explorer",           icon="🔍"),
]
nav = st.navigation(pages)

from llm_chat import render_sidebar_chat

render_sidebar_chat()

nav.run()

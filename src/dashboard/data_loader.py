"""
Dual-mode data loader for the Digital India dashboard.

WEB_MODE (env var) controls the data source:
  - "true"  → reads pre-exported CSVs (for Streamlit Community Cloud)
  - "false" → queries MongoDB locally (default)
"""

import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SQLITE_PATH = ROOT / "db" / "sqlite" / "dsm.db"
DATA_DIR = ROOT / "cleaned_datasets"

WEB_MODE = os.getenv("WEB_MODE", "false").lower() == "true"


def _mongo_db():
    from pymongo import MongoClient
    client = MongoClient("mongodb://localhost:27017")
    return client, client["dsm"]


# ── National wireless subscriber time series ─────────────────────────────

@st.cache_data(ttl=3600)
def load_wireless_ts():
    if WEB_MODE:
        df = pd.read_csv(DATA_DIR / "national_wireless_ts.csv")
        df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
        return df.sort_values("date").reset_index(drop=True)

    from src.analysis.obj1_wireless_growth import get_national_wireless_ts
    client, db = _mongo_db()
    ts = get_national_wireless_ts(db)
    client.close()
    return ts


# ── HHI by state and year ────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_hhi():
    if WEB_MODE:
        return pd.read_csv(DATA_DIR / "hhi_by_state_year.csv")

    from src.analysis.obj1_wireless_growth import compute_hhi
    client, db = _mongo_db()
    hhi = compute_hhi(db)
    client.close()
    return hhi


# ── Provider market shares ────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_provider_shares():
    if WEB_MODE:
        return pd.read_csv(DATA_DIR / "provider_shares.csv", index_col="year")

    from src.analysis.obj1_wireless_growth import get_provider_shares
    client, db = _mongo_db()
    shares = get_provider_shares(db)
    client.close()
    return shares


# ── Digital transactions (SQLite — works in both modes) ───────────────────

@st.cache_data(ttl=3600)
def load_digital_transactions():
    conn = sqlite3.connect(str(SQLITE_PATH))
    df = pd.read_sql_query("SELECT * FROM digital_transactions", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df

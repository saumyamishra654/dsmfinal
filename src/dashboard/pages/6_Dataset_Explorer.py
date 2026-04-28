import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "db" / "sqlite" / "dsm.db"
CSV_PATH = ROOT / "cleaned_datasets" / "telecom_subscriptions.csv"

SQLITE_TABLES = {
    "States (lookup)": "states",
    "Tele-density (monthly, by state)": "tele_density",
    "Wired & Wireless Subscribers (monthly, by state)": "wired_wireless",
    "Education GER (yearly, by state/gender/category)": "education_ger",
    "Digital Transactions (monthly, national)": "digital_transactions",
    "Electricity Consumption (yearly, by sector)": "electricity_consumption",
}


@st.cache_data(ttl=3600)
def load_sqlite_table(table_name):
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df


@st.cache_data(ttl=3600)
def load_telecom_csv():
    return pd.read_csv(CSV_PATH)


st.header("Dataset Explorer")
st.markdown(
    "Browse the raw datasets used in this study. Select a dataset below to view "
    "its contents, column types, and summary statistics."
)

st.divider()

# Dataset selector
all_options = list(SQLITE_TABLES.keys()) + ["Telecom Subscriptions (provider-level, from CSV)"]
choice = st.selectbox("Select a dataset", all_options)

if choice == "Telecom Subscriptions (provider-level, from CSV)":
    df = load_telecom_csv()
    st.caption(
        "Source: TRAI telecom subscription data, cleaned and provider-normalized. "
        "Originally stored in MongoDB; exported to CSV for portability. "
        "Contains per-provider subscriber counts by state and month."
    )
else:
    table_name = SQLITE_TABLES[choice]
    df = load_sqlite_table(table_name)

    descriptions = {
        "states": "Lookup table mapping integer state_id to state_name. "
                  "All other tables reference this via foreign key.",
        "tele_density": "Monthly tele-density (telephones per 100 people) by state. "
                        "Source: TRAI. Period: 2013-2023.",
        "wired_wireless": "Monthly wired and wireless subscriber counts (in millions) by state. "
                          "Source: TRAI. Period: 2013-2023.",
        "education_ger": "Gross Enrolment Ratio by state, gender (Male/Female/Total), "
                         "and category (All Categories/Scheduled Caste/Scheduled Tribe). "
                         "Source: AISHE / Ministry of Education. Period: 2012-2021.",
        "digital_transactions": "Monthly national digital transaction volumes in crores. "
                                "Includes total digital, UPI/BHIM, and debit card breakdowns. "
                                "Source: RBI. Period: 2016-2021.",
        "electricity_consumption": "Yearly electricity consumption by sector (Industry, Domestic, "
                                   "Commercial, etc.) in GWh. Source: CEA. Period: 1970-2023.",
    }
    st.caption(descriptions.get(table_name, ""))

# Info bar
col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Columns", len(df.columns))
col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.0f} KB")

# Filter controls
st.markdown("#### Filter")

filter_col = st.selectbox("Filter by column", ["(no filter)"] + list(df.columns))
if filter_col != "(no filter)":
    unique_vals = df[filter_col].dropna().unique()
    if len(unique_vals) <= 100:
        selected = st.multiselect(f"Values for {filter_col}", sorted(unique_vals, key=str))
        if selected:
            df = df[df[filter_col].isin(selected)]
    else:
        search = st.text_input(f"Search in {filter_col}")
        if search:
            df = df[df[filter_col].astype(str).str.contains(search, case=False, na=False)]

# Data table
st.markdown(f"#### Data ({len(df):,} rows)")
st.dataframe(df, use_container_width=True, hide_index=True, height=400)

# Summary statistics
with st.expander("Summary Statistics"):
    numeric_cols = df.select_dtypes(include="number")
    if not numeric_cols.empty:
        st.dataframe(numeric_cols.describe().round(2), use_container_width=True)
    else:
        st.info("No numeric columns in this dataset.")

# Column info
with st.expander("Column Types"):
    type_df = pd.DataFrame({
        "Column": df.columns,
        "Type": [str(df[c].dtype) for c in df.columns],
        "Non-null": [df[c].notna().sum() for c in df.columns],
        "Null": [df[c].isna().sum() for c in df.columns],
        "Unique": [df[c].nunique() for c in df.columns],
    })
    st.dataframe(type_df, use_container_width=True, hide_index=True)

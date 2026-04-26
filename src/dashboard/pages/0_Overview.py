import streamlit as st

st.header("Digital India: Telecom, Education & Payments")
st.markdown(
    "This study investigates how India's telecom expansion between 2013 and 2023 reshaped "
    "education access, digital payments, and the gap between digitally advanced and lagging states. "
    "We combine five national datasets (telecom subscriber counts, tele-density, gross enrolment "
    "ratios (GER), digital transaction volumes, and electricity consumption) spanning over a decade "
    "of India's digital transformation."
)

st.divider()

# ---------------------------------------------------------------------------
# Research Questions
# ---------------------------------------------------------------------------
st.subheader("Research Questions")

col1, col2 = st.columns(2)
with col1:
    st.info(
        "**RQ 1: Growth & Market Disruption**\n\n"
        "How did India's wireless subscriber base grow between 2013–2023, and were there "
        "identifiable structural breaks driven by policy or market events — most notably "
        "Reliance Jio's entry in September 2016?"
    )
    st.info(
        "**RQ 2: Connectivity & Education**\n\n"
        "Does increased telecom access (tele-density) translate into higher Gross Enrolment "
        "Ratio (GER), i.e., education rates at the state level? Is the effect different for women and SC/ST groups?"
    )
with col2:
    st.info(
        "**RQ 3: Connectivity & the Digital Economy**\n\n"
        "Did wireless subscriber growth affect digital transaction volumes? "
        "Does the co-movement of electricity, wireless, and payment data tell a coherent story "
        "of digital enablement?"
    )
    st.info(
        "**RQ 4: The Digital Divide**\n\n"
        "Which Indian states are digitally excluded? Is there any"
        "cluster of states based on their digital enablement, and "
        "how many years would it take the lagging states to reach the leaders at "
        "their current growth rates?"
    )

st.divider()

# ---------------------------------------------------------------------------
# Key Findings
# ---------------------------------------------------------------------------
st.subheader("Key Findings at a Glance")

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    label="Wireless Subscribers (latest)",
    value="~1.17 B",
    delta="10× growth since 2013",
)
c2.metric(
    label="Structural Breaks Detected",
    value="2",
    delta="2011 mobile explosion · 2016 Jio entry",
)
c3.metric(
    label="Bihar – Years to Close Gap",
    value="~135 yrs",
    delta="at current tele-density growth rate",
    delta_color="inverse",
)
c4.metric(
    label="Education & Digital Enablement positively correlated",
    value="📈",
    delta="states with high telecom density tend to have high GERs",
)

st.divider()

# ---------------------------------------------------------------------------
# Data Sources
# ---------------------------------------------------------------------------
st.subheader("Data Sources")

st.markdown("""
| Dataset | Source | Rows | Period |
|---|---|---|---|
| Area-wise Tele-density | TRAI | 2,100 | 2013–2023 |
| Wired & Wireless Subscribers | TRAI | 1,694 | 2013–2023 |
| Gross Enrolment Ratio (GER) | AISHE / Ministry of Education | 3,231 | 2012–2021 |
| Digital Transactions | RBI | 64 | 2016–2021 |
| Sector-wise Electricity Consumption | CEA | 158 | 1970–2023 |
""")

st.caption(
    "Panel states: 17 major states where TRAI telecom circle names map cleanly to state names. "
    "Metro sub-circles (Kolkata, Mumbai) and composite circles (North East) are excluded to avoid "
    "double-counting or non-attributable aggregation."
)

st.divider()

# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------
st.subheader("Methodology")

col_l, col_r = st.columns(2)
with col_l:
    st.markdown("""
**Data Pipeline**
- Raw CSVs cleaned via `clean.py` → type-preserving `.pkl` intermediate files
- SQLite: 3NF-normalised schema (6 tables, `state_id` surrogate key, FK constraints)
- MongoDB: provider-level monthly subscriber data (market share & HHI analysis)
- Financial Year vs Calendar Year alignment handled at month level during ingestion

**Panel Construction**
- 17 states × 2013–2021 (tele-density joined to GER)
- One-year lag added: tele-density at (t−1) predicts GER at (t)
- 143 observations after dropping the first year per state
""")
with col_r:
    st.markdown("""
**Statistical Methods**
- **Structural break detection:** Bai-Perron algorithm + Chow tests on the national wireless subscriber series
- **Granger causality:** VAR-based F-test (lags 1–4 months) on month-on-month growth rates
- **Panel regression:** Two-way fixed effects (state + year FE), clustered standard errors by state
- **Clustering:** Louvain community detection on a cosine-similarity weighted graph
- **Gap analysis:** Forward projection: years at current annual growth rate to reach leader community mean tele-density
""")

st.divider()

st.caption(
    "Navigate using the sidebar to explore each research question in detail. "
    "Use the **State Explorer** to drill into any individual state's connectivity and education trends."
)

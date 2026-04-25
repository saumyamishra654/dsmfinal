# DSM Project Design Spec
## Digital Infrastructure and Its Outcomes: Telecom, Payments, and Education in India

**Authors:** Saumya Mishra & Vatsl Goswami
**Date:** 2026-04-23

---

## 1. Research Question

How has India's telecom infrastructure expansion — particularly the wireless/mobile revolution — driven digital financial inclusion and higher education access across states?

## 2. Database Architecture

### 2.1 SQLite (Relational) — 5 tables

**`states`** — dimension table solving the telecom-circle-to-state mapping problem
- `state_id` (PK), `state_name`, `telecom_circle`
- Handles splits: "UP East" + "UP West" -> "Uttar Pradesh"; "Maharashtra & Goa" -> separate rows
- All other tables FK to this

**`tele_density`** — from `area-wise tele density.csv` (~2100 rows)
- `id` PK, `state_id` FK, `year` INT, `month` INT, `date` DATE, `tele_density` REAL
- Parsed from "Financial Year (Apr - Mar), 2023" and "November, 2023" format

**`wired_wireless`** — from `wired, wireless telephone.csv` (~1694 rows)
- `id` PK, `state_id` FK, `year` INT, `month` INT, `date` DATE
- `wireline_millions` REAL, `wireless_millions` REAL, `pct_share` REAL

**`education_ger`** — from `education-enrolment.csv` (~3231 rows)
- `id` PK, `state_id` FK, `year` INT, `gender` TEXT, `category` TEXT, `ger` REAL
- 270 missing GER values (mostly UTs) — store as NULL

**`digital_transactions`** — from `digital transactions.csv` (~64 rows)
- `id` PK, `year` INT, `month` INT, `date` DATE
- `digital_txn_crores` REAL, `bhim_txn_crores` REAL, `debit_card_crores` REAL
- National-level only, no state FK

**`electricity_consumption`** — from `sector-wise electricity consumption.csv` (~158 rows)
- `id` PK, `year` INT, `sector` TEXT, `additional_info` TEXT
- `energy_gwh` REAL, `pct_consumption` REAL, `pct_growth` REAL

**Indexes:** Composite index on `(state_id, date)` for tele_density and wired_wireless. Index on `(state_id, year)` for education_ger.

### 2.2 MongoDB — 1 collection

**`telecom_subscriptions`** — from `telecom subscription data.csv` (~58K docs)
```json
{
  "state": "Andaman And Nicobar Islands",
  "telecom_circle": "West Bengal",
  "provider": "BSNL",
  "year": 2021, "month": 4,
  "wireless_subscribers": 10519.48,
  "wireline_subscribers": 825.19,
  "vlr_proportion": 0.30
}
```
- Nullable fields stored as absent (not null) — MongoDB's natural sparse representation
- Wireless ~11% missing, Wireline ~56% missing, VLR ~35% missing
- Indexes: `{ state: 1, year: 1, month: 1 }`, `{ provider: 1 }`

### 2.3 Why this split

| Criterion | SQL (SQLite) | MongoDB |
|-----------|-------------|---------|
| Schema | Fixed, clean columns | Sparse, variable per provider |
| Missing data | Rare (0-8%) | Heavy (11-56%) |
| Key operations | JOINs (state mapping), aggregation | Provider-level grouping, flexible filtering |
| Course concept | Relational model, normalization, SQL queries (ch2/ch3) | Document store, aggregation pipeline, schema flexibility (NoSQL lectures) |

## 3. Analytical Objectives

### Objective 1 — Quantify wireless growth + identify Jio structural break (2008-2021)

**Data source:** MongoDB `telecom_subscriptions` (provider-level, 2008-2021)

**Analyses:**
1. **MongoDB aggregation pipeline:** `$group` by state + year + month to get total wireless subscribers nationally. This becomes the main time-series.
2. **Bai-Perron endogenous structural break detection** on the national wireless subscriber series — let the algorithm find the break date rather than hardcoding Jan 2016. Confirm with Chow test.
3. **Pre-Jio vs post-Jio CAGR** — compute compound annual growth rate before and after the detected break.
4. **State-level growth rates** pre/post break, ranked — identifies which states Jio lifted most.
5. **HHI (Herfindahl-Hirschman Index)** per state per year — computed via MongoDB `$group` by state+year, then sum of squared market shares. Shows market concentration dropping when Jio enters.
6. **Stacked area chart of provider market share** — Jio eating into Airtel/Vodafone/BSNL over time.

**Visualizations:** National wireless subscriber line chart with break annotation, state-level growth ranking bar chart, HHI over time, provider market share stacked area.

### Objective 2 — Tele-density vs GER correlation, and did it change post-Jio? (2013-2021)

**Data source:** SQLite `tele_density` JOIN `education_ger` via `states` (panel: state x year, 2013-2021)

**Analyses:**
1. **SQL JOIN** tele_density with education_ger through states table — demonstrate relational algebra from ch2/ch3.
2. **Pearson + Spearman correlation per year**, plotted as a line — shows whether relationship strengthened post-Jio.
3. **Panel regression with two-way fixed effects:** `GER_it = alpha + beta * tele_density_it + state_FE + year_FE + epsilon` using `linearmodels.PanelOLS`.
4. **Interaction term** `beta * post_Jio_dummy` — directly tests whether the slope changed after 2016.
5. **Lagged effects:** Test `GER_it = beta * tele_density_{i,t-1}` and `t-2`. If lagged coefficient is stronger, that's a causal-direction argument.
6. **Quantile regression** (`statsmodels.QuantReg`) — does telecom matter more for states at the bottom of the GER distribution?
7. **Disaggregated runs:** Separate regressions for Female GER and SC/ST GER — tests differential impact of telecom on marginalized groups.

**Visualizations:** Scatter plots (tele-density vs GER) colored by pre/post Jio, correlation-over-time line, regression coefficient forest plot, quantile regression coefficient plot.

### Objective 3 — Digital transactions growth + debit-to-UPI shift (2017-2022)

**Data source:** SQLite `digital_transactions` (national, 64 rows)

**Analyses:**
1. **Derived metric:** `UPI_share = (digital_txn - debit_card) / digital_txn` over time.
2. **STL decomposition** (seasonal-trend using LOESS) on total digital transaction series — separate trend from seasonality.
3. **Stacked area chart:** BHIM/UPI share vs debit-card share — decomposes the shift visually.
4. **Granger causality test** (national level): does wireless subscriber growth predict digital txn growth? Uses `statsmodels.tsa.stattools.grangercausalitytests`.
5. **COVID event study:** Visible acceleration around Mar 2020 lockdown.
6. **Explicit caveat** in report: cannot do state-level panel because dataset 2.4 is national only.

**Visualizations:** Total digital txn time series with STL decomposition, stacked area of payment method shares, Granger causality lag plot.

### Objective 4 — Identify digitally excluded states (digital divide clustering)

**Data source:** SQLite `tele_density` + `education_ger` via `states` (state x year)

**Analyses:**
1. **Feature engineering per state:** mean_tele_density, tele_density_slope, mean_GER, GER_slope (4 features per state).
2. **K-means clustering** with silhouette score to select k. StandardScaler before clustering.
3. **Louvain community detection** on a state-similarity graph: nodes = states, edges weighted by cosine similarity of the 4-feature vectors. Connects to SNA coursework.
4. **Compare K-means vs Louvain** — if they agree, clustering is robust. If not, discuss why.
5. **PCA biplot** — shows what drives cluster separation (level vs trajectory).
6. **Gap analysis:** For each "laggard" state, compute how many years behind the "leader" cluster it is on the tele-density trajectory. E.g., "Bihar in 2021 had the tele-density that Maharashtra had in 2015."

**Visualizations:** Choropleth map of cluster membership, PCA biplot, gap analysis timeline.

### Objective 5 — Electricity consumption as corroboration proxy

**Data source:** SQLite `electricity_consumption` (national, sectoral, 1970-2023)

**Analyses:**
1. **Overlay normalized time series:** Domestic+Commercial electricity, wireless subs, digital txn — all should show post-2010 hockey-stick.
2. **Cross-correlation function (CCF)** between domestic electricity growth and wireless subscriber growth at different lags — quantifies whether electrification leads or follows digital adoption.
3. **Growth-rate acceleration test:** Is domestic electricity growth rate structurally higher post-2010?

**Visualizations:** Overlaid normalized series, CCF plot.

## 4. Data Cleaning Pipeline

### 4.1 Date normalization
- Parse "Financial Year (Apr - Mar), 2023" → extract year as integer
- Parse "November, 2023" → extract month as integer
- Construct proper DATE column for time-series ordering
- Align FY vs CY: FY 2023 month "April" = CY 2022-04

### 4.2 Telecom circle to state mapping
- Build explicit mapping dict: `{"UP East": "Uttar Pradesh", "UP West": "Uttar Pradesh", "Mumbai": "Maharashtra", ...}`
- For circles spanning multiple states (Maharashtra & Goa): keep as the circle name in the `states` table and note the limitation in the report. Do not attempt population-weighted splitting — the tele-density is already per-100-people so the circle-level number is usable as-is.
- Populate `states` table first, then FK all other tables

### 4.3 Missing value handling
- Education GER: 270 NULLs in UTs — exclude UTs from state-level panel analysis, note in report
- Telecom subscriptions (MongoDB): absent fields = missing. In aggregation pipeline, use `$ifNull` or filter with `$exists`
- Electricity growth rates: some missing — interpolate for visualization, exclude from statistical tests

### 4.4 Provider name normalization (MongoDB)
- Standardize provider names: "Bharti" and "Bharti Airtel (including Tata Tele.)" need mapping
- Create a provider_mapping dict applied during ingestion

## 5. Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.11+ |
| SQL database | SQLite via `sqlite3` stdlib |
| NoSQL database | MongoDB via `pymongo` |
| Data manipulation | pandas, numpy |
| Statistical analysis | scipy, statsmodels, linearmodels |
| ML / clustering | scikit-learn |
| Graph / community detection | networkx, community (python-louvain) |
| Visualization | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |
| LLM interface | Anthropic Claude API |
| Structural break | `ruptures` library (Bai-Perron) |

## 6. Project Structure

```
dsmfinal/
  datasets/               # raw CSVs (already present)
  docs/                   # course PDFs + specs
  src/
    data/
      clean.py            # date parsing, name normalization, mapping dicts
      load_sqlite.py      # create schema + ingest into SQLite
      load_mongo.py       # ingest provider-level data into MongoDB
    analysis/
      obj1_wireless_growth.py    # Bai-Perron, HHI, CAGR, provider shares
      obj2_teledensity_ger.py    # panel regression, quantile reg, lag analysis
      obj3_digital_txn.py        # STL, Granger, UPI share
      obj4_clustering.py         # K-means, Louvain, PCA, gap analysis
      obj5_electricity.py        # CCF, normalized overlay
    dashboard/
      app.py              # Streamlit app
      pages/              # multi-page Streamlit layout
    llm/
      agent.py            # Claude API-powered dataset query interface
  notebooks/
    eda.ipynb             # exploratory data analysis notebook
  db/
    dsm.db                # SQLite database file (gitignored)
  requirements.txt
  README.md
```

## 7. Bonus Components

### 7.1 Streamlit Dashboard

Multi-page Streamlit app with Plotly interactive charts:

**Page 1 — National Overview:**
- Wireless subscriber growth (with break annotation)
- Digital transaction trend with STL decomposition
- Provider market share stacked area

**Page 2 — State Explorer:**
- State dropdown selector
- Tele-density time series for selected state
- GER time series for selected state (by gender/category toggle)
- Side-by-side comparison mode (2 states)

**Page 3 — Digital Divide Map:**
- Choropleth of India colored by cluster membership
- Year slider to see evolution
- Click a state to see its feature profile

**Page 4 — Correlations & Regression:**
- Interactive scatter: tele-density vs GER, year slider
- Regression results table
- Quantile regression coefficient plot

### 7.2 LLM Query Interface

Claude API-powered natural-language interface for querying the datasets:

- User types a question like "Which state had the highest GER growth after 2016?"
- System sends the question + schema context to Claude API
- Claude generates a SQL query or MongoDB aggregation
- System executes the query against the actual databases
- Claude interprets the results in natural language
- Built as a Streamlit page (Page 5) or standalone script

**Implementation:**
- System prompt includes table schemas, column descriptions, sample values
- Tool use: Claude generates SQL/MongoDB queries as tool calls
- Execute against SQLite/MongoDB, return results
- Claude summarizes the answer

## 8. Report Structure (for submission)

1. Problem Statement & Objectives
2. Dataset Descriptions
3. Database Design — schema diagrams, SQL DDL, MongoDB document structure, justification for SQL vs MongoDB split
4. Data Cleaning & Integration — date normalization, circle-to-state mapping, missing values
5. Exploratory Data Analysis — distributions, patterns, initial visualizations
6. Analysis & Findings — one section per objective (1-5), each with method, results, interpretation
7. Recommendations — policy suggestions grounded in findings
8. Bonus Components — dashboard screenshots, LLM interface demo
9. Limitations & Future Work

## 9. Work Split Suggestion

The project naturally splits into two independent tracks after the shared data pipeline:

**Track A (Data + SQL + Objectives 2,4):**
- Data cleaning pipeline
- SQLite schema + ingestion
- Objective 2 (panel regression, quantile reg)
- Objective 4 (clustering, community detection)
- Dashboard Pages 2, 3, 4

**Track B (MongoDB + Objectives 1,3,5 + LLM):**
- MongoDB ingestion + provider normalization
- Objective 1 (structural break, HHI, provider analysis)
- Objective 3 (digital transactions, Granger)
- Objective 5 (electricity corroboration)
- Dashboard Pages 1, 5 (LLM interface)

Both tracks can work in parallel after the shared `states` mapping table and `clean.py` are done.

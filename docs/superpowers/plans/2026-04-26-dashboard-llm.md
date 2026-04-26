# Dashboard + LLM Query Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-page Streamlit dashboard with interactive Plotly charts for all analysis objectives, plus an LLM-powered query interface in the sidebar that generates and runs Python code against the databases.

**Architecture:** Multi-page Streamlit app (`st.navigation`). Each page loads data from SQLite/MongoDB via cached functions and renders Plotly charts. The sidebar hosts a LangChain-powered chat that sends user questions to Claude, receives Python code, evaluates it dynamically (locally hosted one-time demo, user-requested), and displays results with a "Show Code" expander.

**Tech Stack:** Streamlit, Plotly, LangChain + langchain-anthropic, pandas, pymongo, sqlite3

---

### Task 1: Install dependencies and create directory structure

**Files:**
- Create: `src/dashboard/app.py`
- Create: `src/dashboard/pages/` (directory)
- Create: `src/dashboard/llm_chat.py`

- [ ] **Step 1: Install packages**

```bash
pip install streamlit langchain langchain-anthropic plotly
```

- [ ] **Step 2: Create directory structure**

```bash
mkdir -p src/dashboard/pages
```

- [ ] **Step 3: Create `app.py` skeleton**

Write `src/dashboard/app.py` with page navigation setup and sidebar chat import. Entry point for `streamlit run src/dashboard/app.py`.

Pages registered via `st.Page()`:
1. `pages/1_National_Overview.py` — National Overview
2. `pages/2_State_Explorer.py` — State Explorer
3. `pages/3_Digital_Divide.py` — Digital Divide
4. `pages/4_Analysis_Results.py` — Analysis Results

Import and call `render_sidebar_chat()` from `llm_chat.py` to render the LLM interface in the sidebar on every page.

Config: `st.set_page_config(page_title="Digital India Dashboard", page_icon="📡", layout="wide", initial_sidebar_state="expanded")`

- [ ] **Step 4: Commit**

```bash
git add src/dashboard/
git commit -m "feat: scaffold dashboard app with page navigation"
```

---

### Task 2: LLM sidebar chat (`llm_chat.py`)

**Files:**
- Create: `src/dashboard/llm_chat.py`

- [ ] **Step 1: Write `llm_chat.py`**

Core module with these components:

**Constants:**
- `DB_PATH` = path to `db/sqlite/dsm.db`
- `MONGO_URI` = `"mongodb://localhost:27017"`
- `MONGO_DB` = `"dsm"`

**`SYSTEM_PROMPT`** — instructs Claude to return ONLY a Python code block that assigns output to `result`. Lists the full SQLite schema (all 6 tables with columns and types) and MongoDB collection schema. Specifies available namespace: `pd`, `np`, `sqlite3`, `pymongo`, `px` (plotly.express), `go` (plotly.graph_objects), `db_path`, `mongo_uri`, `mongo_db`.

**`QUICK_QUERIES`** — list of 4 strings:
- "Top 5 states by tele-density in 2021"
- "Compare wireless growth: Bihar vs Maharashtra"
- "Monthly UPI share trend"
- "Which provider grew fastest after 2016?"

**`_extract_code(text: str) -> str`** — regex to pull Python code from markdown code fences (` ```python ... ``` ` or ` ``` ... ``` `). Falls back to raw text if no fence found.

**`_run_code(code: str) -> tuple[result, error_or_None]`** — builds a namespace dict with pd, np, sqlite3, pymongo, px, go, db_path, mongo_uri, mongo_db. Uses Python's built-in dynamic code evaluation to run the generated code. Returns the `result` variable from the namespace, or an error traceback string. Note: this is intentionally unsafe (user-requested for a locally hosted one-time demo).

**`_ask_llm(question: str, api_key: str) -> str`** — creates `ChatAnthropic(model="claude-sonnet-4-20250514", api_key=api_key, max_tokens=2048)`, sends `[SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]`, returns `response.content`.

**`render_sidebar_chat()`** — renders in `st.sidebar`:
1. API key `st.text_input(type="password")`
2. Quick query buttons in 2-column grid
3. Chat input via `st.chat_input()` (outside sidebar since Streamlit requires it in main area)
4. On query: call `_ask_llm` → `_extract_code` → `_run_code`
5. Store `{question, code, result, error}` in `st.session_state.chat_history`
6. Display history: DataFrame → `st.dataframe()`, plotly Figure → `st.plotly_chart()`, else → `st.write()`. Each entry has `st.expander("Show generated code")` with `st.code(code, language="python")`.
7. If no API key: show `st.info("Enter your Anthropic API key above to use AI queries.")`

- [ ] **Step 2: Verify import works**

```bash
python -c "import sys; sys.path.insert(0,'src/dashboard'); from llm_chat import render_sidebar_chat; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/dashboard/llm_chat.py
git commit -m "feat: add LLM sidebar chat with code evaluation and Show Code expander"
```

---

### Task 3: Page 1 — National Overview

**Files:**
- Create: `src/dashboard/pages/1_National_Overview.py`

- [ ] **Step 1: Write Page 1**

Page content (top to bottom):

**Header:** `st.header("📊 National Overview")`

**Row 1 — Key metrics** (3 columns):
- Total wireless subscribers (latest from MongoDB via `get_national_wireless_ts`)
- Structural breaks detected ("2" with dates)
- UPI share (latest) with delta from earliest

**Row 2 — Two columns:**
- Left: Wireless subscriber growth line chart (Plotly `px.line`) with two `add_vline` annotations (Jan 2011 "Mobile Explosion" orange, Jun 2016 "Jio Entry" red)
- Right: Provider market share stacked area (`go.Scatter` with `stackgroup="one"`) from `get_provider_shares()`

**Row 3 — Two columns:**
- Left: Digital transaction volume line chart (Plotly `px.line`, green) with COVID vline at 2020-03-01
- Right: Payment composition stacked area (UPI/BHIM, Debit Card, Other Digital) from SQLite digital_transactions

**Row 4 — Full width:**
- HHI over time line chart with horizontal reference lines at 1500 (orange "Moderately concentrated") and 2500 (red "Highly concentrated")

**Data loading:** Use `@st.cache_data(ttl=3600)` wrapper functions that call obj1 functions (`get_national_wireless_ts`, `detect_structural_breaks`, `compute_hhi`, `get_provider_shares`) and SQLite queries.

- [ ] **Step 2: Commit**

```bash
git add src/dashboard/pages/1_National_Overview.py
git commit -m "feat: add National Overview dashboard page"
```

---

### Task 4: Page 2 — State Explorer

**Files:**
- Create: `src/dashboard/pages/2_State_Explorer.py`

- [ ] **Step 1: Write Page 2**

**Controls:**
- `st.toggle("Compare two states")` — toggles single vs dual state mode
- State dropdown(s) — populated from `SELECT DISTINCT state_name FROM states JOIN tele_density`
- `st.radio("GER Gender", ["Total", "Male", "Female"], horizontal=True)`
- `st.radio("GER Category", ["All Categories", "Scheduled Caste", "Scheduled Tribe"], horizontal=True)`

**Charts (for each selected state):**
1. Tele-density time series — `go.Scatter` line, one trace per state (overlaid in compare mode)
2. Wired vs Wireless — two `go.Scatter` traces (wireless blue, wireline orange), one chart per state side-by-side in compare mode
3. GER — `go.Scatter` lines+markers, filtered by selected gender and category

**Data loading:** `@st.cache_data` functions with parameterized SQL queries (JOIN states table, WHERE state_name = ?).

- [ ] **Step 2: Commit**

```bash
git add src/dashboard/pages/2_State_Explorer.py
git commit -m "feat: add State Explorer page with comparison mode"
```

---

### Task 5: Page 3 — Digital Divide (Placeholder)

**Files:**
- Create: `src/dashboard/pages/3_Digital_Divide.py`

- [ ] **Step 1: Write Page 3 placeholder**

- `st.info("🚧 Clustering analysis pending — will show choropleth map of digital divide clusters (K-means + Louvain) once Objective 4 is complete.")`
- Horizontal bar chart (`px.bar` with `orientation="h"`) of states ranked by latest tele-density, colored by value (RdYlGn scale)
- Summary stats in side column: highest, lowest, median, count above 100

**Data:** SQL query for latest month of latest year in tele_density, joined with states.

- [ ] **Step 2: Commit**

```bash
git add src/dashboard/pages/3_Digital_Divide.py
git commit -m "feat: add Digital Divide placeholder page with state rankings"
```

---

### Task 6: Page 4 — Analysis Results

**Files:**
- Create: `src/dashboard/pages/4_Analysis_Results.py`

- [ ] **Step 1: Write Page 4**

**Section 1 — Structural Break Detection:**
- Chow test results table (Break Date, F-statistic, p-value, Significant)
- CAGR by period table (Period label, CAGR %)
- Both computed via cached functions calling `detect_structural_breaks`, `chow_test`, `compute_cagr` from obj1

**Section 2 — Granger Causality:**
- Results table (Lag, F-statistic, p-value, Significant)
- `st.warning()` if no significant results: "Wireless growth does not statistically predict digital transaction growth at the monthly level."
- Computed by merging SQLite wireless + digital_txn, running `grangercausalitytests`

**Section 3 — Electricity Corroboration:**
- Normalized overlay chart (Plotly `go.Figure` with 3 `go.Scatter` traces: electricity, wireless, digital txn)
- Vertical annotations at 2010 and 2016
- x-axis range 1990-2023

**Data loading:** All via `@st.cache_data` functions. Import obj1 functions for break analysis. Granger uses statsmodels directly.

- [ ] **Step 2: Commit**

```bash
git add src/dashboard/pages/4_Analysis_Results.py
git commit -m "feat: add Analysis Results page with break stats, Granger, electricity overlay"
```

---

### Task 7: Smoke test the full app

- [ ] **Step 1: Start the Streamlit app**

```bash
streamlit run src/dashboard/app.py --server.headless true
```

Expected: App starts, no import errors. Visit http://localhost:8501.

- [ ] **Step 2: Verify each page loads**

Navigate to each page in the browser:
- Page 1: Should show wireless growth chart, provider shares, digital txn, HHI
- Page 2: Should show state dropdown, tele-density/wireless/GER charts
- Page 3: Should show placeholder info box + tele-density ranking bar chart
- Page 4: Should show Chow test table, CAGR table, Granger table, electricity overlay

- [ ] **Step 3: Test the LLM sidebar**

1. Enter an API key in the sidebar
2. Click "Top 5 states by tele-density in 2021" quick query button
3. Verify: result appears as a DataFrame, "Show generated code" expander works
4. Type a custom query: "What is the average wireless subscribers per state?"
5. Verify: result appears, code is visible

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete dashboard with 4 pages + LLM sidebar query interface"
```

---

## Verification Checklist

- [ ] `streamlit run src/dashboard/app.py` launches without errors
- [ ] All 4 pages render charts from live data (not static PNGs)
- [ ] LLM sidebar is visible on every page
- [ ] Quick query buttons work (with valid API key)
- [ ] Custom text queries work
- [ ] "Show generated code" expander shows the Python code for each query
- [ ] Page 3 has placeholder message for Vatsl's clustering work
- [ ] Charts use consistent color palette (blue/orange/green/red)

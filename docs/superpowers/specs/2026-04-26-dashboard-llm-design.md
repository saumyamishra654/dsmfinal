# Dashboard + LLM Query Interface — Design Spec

**Date:** 2026-04-26

---

## 1. Overview

Single Streamlit app serving 4 interactive dashboard pages + an AI query chat in the left sidebar (accessible from every page). All charts rendered with Plotly for interactivity. LLM uses LangChain with ChatAnthropic (swappable to other providers).

## 2. File Structure

```
src/dashboard/
  app.py                       # entry point: streamlit run src/dashboard/app.py
  pages/
    1_National_Overview.py     # Obj 1+3: wireless growth, provider shares, digital txn
    2_State_Explorer.py        # Tele-density + GER per state, comparison mode
    3_Digital_Divide.py        # Placeholder for Vatsl's Obj 4 clustering
    4_Analysis_Results.py      # Structural break stats, Granger, electricity overlay, CCF
  llm_chat.py                 # Sidebar chat logic (imported by app.py)
```

## 3. Dashboard Pages

### Page 1 — National Overview
- Wireless subscriber growth line chart (Plotly) with 2 break annotations (Jan 2011, Jun 2016)
- Provider market share stacked area chart
- Digital transaction trend line
- Payment composition stacked area (UPI vs Debit vs Other)
- HHI over time line chart

Data sources: MongoDB `telecom_subscriptions` (via obj1 functions), SQLite `digital_transactions`.

### Page 2 — State Explorer
- State dropdown selector (18 panel states)
- Year range slider
- Tele-density time series for selected state (Plotly line)
- Wired vs wireless subscriber comparison
- GER by gender toggle (Male/Female/Total) and category toggle (All/SC/ST)
- "Compare" mode: second state dropdown, side-by-side charts

Data sources: SQLite `tele_density`, `wired_wireless`, `education_ger` via `states` table.

### Page 3 — Digital Divide (Placeholder)
- Simple state-level tele-density table ranked by latest year
- `st.info("Clustering analysis pending — will show choropleth map of digital divide clusters.")`
- Vatsl fills in with K-means/Louvain results and choropleth when Obj 4 is done

Data source: SQLite `tele_density`.

### Page 4 — Analysis Results
- Structural break summary: detected dates, Chow test F/p, three-period CAGR table
- Granger causality results table (lag, F-stat, p-value, significance)
- Electricity normalized overlay (Plotly, 3 series)
- CCF bar chart
- Key findings as `st.metric` cards

Data sources: Re-run lightweight computations from obj1/obj3/obj5 functions, or cache results.

## 4. LLM Chat (Left Sidebar)

### Location
`st.sidebar` — below the page navigation. Visible and usable from every page.

### Components
1. **Quick-query buttons** — 4 pre-built queries as `st.button`:
   - "Top 5 states by tele-density in 2021"
   - "Compare wireless growth: Bihar vs Maharashtra"
   - "Monthly UPI share trend"
   - "Which provider grew fastest after 2016?"
2. **Text input** — `st.chat_input("Ask a question about the data...")`
3. **Chat history** — stored in `st.session_state.chat_history`, persists across page switches
4. **Result display** — each response shows:
   - The LLM's natural language answer
   - A **"Show Code" expander** (`st.expander("Show generated code")`) revealing the Python code that was run
   - If result is a DataFrame: `st.dataframe()`
   - If result is a plot: `st.plotly_chart()`
   - If result is a scalar/string: `st.write()`

### LLM Implementation

**Provider:** LangChain `ChatAnthropic` (claude-sonnet-4-20250514)
**Swappable:** Change to `ChatOpenAI` or `ChatGoogleGenerativeAI` by changing one import + model name

**System prompt** includes:
- SQLite schema (all 6 tables with column types)
- MongoDB collection schema (telecom_subscriptions fields)
- Instructions: "Return ONLY a Python code block. The code must assign its final result to a variable called `result`. You have access to: `pd` (pandas), `np` (numpy), `sqlite3`, `pymongo`, `plotly.express as px`, `plotly.graph_objects as go`. SQLite connection is at `db_path`. MongoDB is at `mongodb://localhost:27017` database `dsm`."

**Code execution flow:**
1. User types question (or clicks quick-query button)
2. LangChain sends question + system prompt to Claude
3. Extract Python code block from response
4. Run the code in an isolated namespace pre-loaded with:
   - `pd`, `np`, `sqlite3`, `pymongo`, `px` (plotly.express), `go` (plotly.graph_objects)
   - `db_path` pointing to SQLite DB
   - `mongo_uri` and `mongo_db` name
5. Read `result` variable from the namespace
6. Display result based on type (DataFrame -> dataframe, Figure -> plotly_chart, else -> write)
7. Store code + result in `st.session_state.chat_history`

**Note on safety:** This uses Python dynamic code evaluation which is intentionally unsafe. This is acceptable here because: (a) the app is locally hosted, (b) the user explicitly requested this approach for a one-time demo, (c) the code runs with the same permissions as the user. Not appropriate for production.

**API key:** Read from `st.text_input(type="password")` in the sidebar, stored in `st.session_state`. No .env files.

## 5. Styling

- `st.set_page_config(page_title="Digital India Dashboard", layout="wide", page_icon="📡")`
- Consistent color palette across all charts: blue (#1f77b4), orange (#ff7f0e), green (#2ca02c), red (#d62728)
- `st.metric` cards for key statistics on each page
- Clean section headers with `st.header` / `st.subheader`

## 6. Data Loading Strategy

- Use `@st.cache_data` for SQLite queries (cache invalidation on app restart)
- Use `@st.cache_resource` for MongoDB connection
- Import `get_national_wireless_ts` from `obj1_wireless_growth.py` for MongoDB data
- All other data loaded via direct SQLite queries (lightweight, <4K rows)

## 7. Dependencies to Install

```
pip install streamlit langchain langchain-anthropic plotly
```

## 8. Run Command

```bash
streamlit run src/dashboard/app.py
```

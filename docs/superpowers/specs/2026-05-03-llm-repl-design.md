# LLM REPL Chat Integration — Design Spec

## Overview

An AI-powered data analysis chat embedded in the report site. Users ask natural-language questions about India's telecom/education/payments data, an LLM generates Python code, the backend runs it against SQLite, and results (tables, charts, text) are returned to the frontend.

## Architecture

```
Vercel (Next.js frontend)
  Chat component (React):
  - Text input + send button
  - Conversation history (stored in React state)
  - API key input (stored in localStorage, never sent to Vercel)
  - Renders results: DataFrames as tables, Plotly as interactive
    charts, text as paragraphs, errors in red
  - Sends POST to Render with {question, api_key, history}
  - Shows spinner while waiting
  - Quick-query buttons for demo purposes
        |
        | POST /api/chat
        v
Render (FastAPI backend)
  POST /api/chat
  1. Receives {question, api_key, history}
  2. Builds messages: system_prompt + history + user question
  3. Calls Anthropic API (using user's api_key)
  4. Extracts Python code from response (regex on fenced block)
  5. Validates code (blocklist check for dangerous patterns)
  6. Runs code in sandboxed namespace with 10s timeout
  7. Serializes result based on type detection
  8. Returns {code, result_type, data, error}

  Namespace available to generated code:
    pd, np, sqlite3, px, go, db_path

  SQLite: opened read-only (uri mode=ro)
  No MongoDB in production (data pre-flattened into SQLite)
```

## API Contract

### Request

```
POST /api/chat
Content-Type: application/json

{
  "question": "Top 5 states by tele-density in 2021",
  "api_key": "sk-ant-...",
  "history": [
    {"role": "user", "content": "Show me Bihar's wireless growth"},
    {"role": "assistant", "content": "python code here"}
  ]
}
```

- `history`: last 3 turns max (to stay within context without ballooning tokens)
- `api_key`: passed per-request, never stored server-side

### Response

```json
{
  "code": "con = sqlite3.connect(db_path)\ndf = pd.read_sql(...)\nresult = df.head(5)",
  "result_type": "dataframe",
  "data": [{"state": "Delhi", "tele_density": 269.1}],
  "error": null
}
```

**Result types:**

| result_type | data format | Frontend rendering |
|-------------|-------------|-------------------|
| dataframe | JSON array of row objects | table or data grid |
| plotly | Full Plotly JSON spec ({data, layout}) | Plot via react-plotly.js |
| text | String | paragraph |
| error | Error message string | Red error box |

## System Prompt

Same as existing (from llm_chat.py) but updated:
- Remove MongoDB references
- Add the telecom_subscriptions SQLite table (flattened from Mongo)
- Keep schema documentation for all tables
- Keep the "assign to result" convention

## Execution Sandbox

**Allowed in namespace:** pd, np, sqlite3, px, go, db_path

**Blocked patterns (string-matched before running):**
- Dangerous imports (subprocess, shutil, socket, etc.)
- File system access functions
- Network access functions
- Code generation/evaluation functions

**Timeout:** 10 seconds (via signal.alarm on Unix)

**SQLite:** Opened with read-only URI mode to prevent writes.

## Frontend Chat Component

**State:**
- messages: Array of {role, content, code?, resultType?, data?, error?}
- apiKey: string (persisted in localStorage)
- isLoading: boolean

**Quick queries** (pre-populated buttons):
- "Top 5 states by tele-density in 2021"
- "Compare wireless growth: Bihar vs Delhi"
- "Monthly UPI share trend"
- "Show provider market share in 2020"

**UX flow:**
1. User enters API key (stored in localStorage, shown as masked input)
2. User types question or clicks quick-query button
3. Spinner appears
4. Response arrives - rendered below the question
5. Code is shown in a collapsible "Show code" section
6. Conversation scrolls to bottom

## Data Migration: MongoDB to SQLite

Add a telecom_subscriptions table to SQLite:

```sql
CREATE TABLE telecom_subscriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state_id INTEGER REFERENCES states(state_id),
    provider TEXT,
    year INTEGER,
    month INTEGER,
    wireless_subscribers REAL,
    wireline_subscribers REAL
);
```

Populated at deploy time from the existing MongoDB export or from a one-time migration script.

## Files to Create/Modify

**Backend (Render):**
- backend/main.py — FastAPI app with /api/chat + existing data endpoints
- backend/chat.py — LLM call, code extraction, sandbox runner
- backend/sandbox.py — Blocklist validation + timeout runner
- backend/requirements.txt — fastapi, uvicorn, pandas, numpy, plotly, anthropic
- backend/db/dsm.db — SQLite database (copied from project)
- backend/migrate_mongo.py — One-time script to flatten Mongo into SQLite

**Frontend (Vercel):**
- frontend/src/components/Chat.tsx — Main chat component
- frontend/src/components/ChatMessage.tsx — Individual message renderer
- frontend/src/components/PlotlyChart.tsx — Plotly wrapper
- frontend/src/components/DataTable.tsx — DataFrame renderer
- frontend/src/app/explorer/page.tsx — Page that hosts the chat

## Non-Goals

- No authentication/user accounts
- No conversation persistence across page reloads (React state only)
- No streaming (simple request/response)
- No multi-model support (Claude only)
- No production-grade sandboxing (Docker, gVisor, etc.)

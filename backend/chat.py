"""LLM chat handler for data analysis queries."""

import re
from pathlib import Path

import anthropic

from sandbox import validate_code, run_sandboxed

DB_PATH = str(Path(__file__).parent / "db" / "dsm.db")

SYSTEM_PROMPT = """You are a data analyst assistant with access to a SQLite database about India's digital infrastructure.

The database is located at the path stored in `db_path`. You have access to: pd, np, sqlite3, px (plotly.express), go (plotly.graph_objects), and db_path.

## Database Schema

### states
- state_id INTEGER PRIMARY KEY
- state_name TEXT

### tele_density
- state_id INTEGER (FK -> states)
- year INTEGER
- month INTEGER
- tele_density REAL

### wired_wireless
- state_id INTEGER (FK -> states)
- year INTEGER
- month INTEGER
- wireless_millions REAL
- wireline_millions REAL

### education_ger
- state_id INTEGER (FK -> states)
- year INTEGER
- gender TEXT (Male/Female/Total)
- category TEXT (Primary/Upper Primary/Secondary/Higher Secondary/Higher Education)
- ger REAL

### digital_transactions
- year INTEGER
- month INTEGER
- volume_crore REAL
- value_lakh_crore REAL

### electricity_consumption
- state_id INTEGER (FK -> states)
- year INTEGER
- consumption_gwh REAL

### telecom_subscriptions
- state_id INTEGER (FK -> states)
- year INTEGER
- month INTEGER
- provider TEXT
- subscribers_millions REAL

## Instructions

1. Write Python code to answer the user's question.
2. Use `sqlite3.connect(db_path)` to connect to the database.
3. Use pandas for data manipulation and plotly for visualizations.
4. ALWAYS assign your final answer to a variable called `result`.
   - If the answer is a DataFrame, assign the DataFrame to `result`.
   - If the answer is a plot, assign the plotly Figure to `result`.
   - If the answer is a simple value/text, assign a string to `result`.
5. Wrap your code in a ```python ... ``` block.
6. Keep code concise and efficient.
"""


def _extract_code(text: str) -> str | None:
    """Extract Python code from markdown code blocks."""
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _serialize_result(result) -> tuple[str, any]:
    """Serialize the execution result for JSON response.

    Returns:
        Tuple of (type_str, data) where type_str is one of
        "dataframe", "plotly", "text", or "error".
    """
    import pandas as pd
    import plotly.graph_objects as go

    if result is None:
        return ("text", "No result produced.")

    if isinstance(result, pd.DataFrame):
        return ("dataframe", result.to_dict(orient="records"))

    if isinstance(result, go.Figure):
        return ("plotly", result.to_json())

    return ("text", str(result))


def handle_chat(question: str, api_key: str, history: list[dict] | None = None) -> dict:
    """Handle a chat request by calling the LLM, extracting code, and executing it.

    Args:
        question: The user's question.
        api_key: Anthropic API key.
        history: Optional conversation history as list of {role, content} dicts.

    Returns:
        Dict with keys: code, result_type, data, error
    """
    messages = []

    if history:
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
    except anthropic.AuthenticationError:
        return {
            "code": None,
            "result_type": "error",
            "data": None,
            "error": "Invalid API key. Please check your Anthropic API key.",
        }
    except Exception as e:
        return {
            "code": None,
            "result_type": "error",
            "data": None,
            "error": f"LLM API error: {e}",
        }

    assistant_text = response.content[0].text
    code = _extract_code(assistant_text)

    if not code:
        return {
            "code": None,
            "result_type": "text",
            "data": assistant_text,
            "error": None,
        }

    validation_error = validate_code(code)
    if validation_error:
        return {
            "code": code,
            "result_type": "error",
            "data": None,
            "error": validation_error,
        }

    result, exec_error = run_sandboxed(code, DB_PATH)

    if exec_error:
        return {
            "code": code,
            "result_type": "error",
            "data": None,
            "error": exec_error,
        }

    result_type, data = _serialize_result(result)

    return {
        "code": code,
        "result_type": result_type,
        "data": data,
        "error": None,
    }

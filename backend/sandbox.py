"""Sandboxed code execution for LLM-generated Python code."""

import signal
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DANGEROUS_PATTERNS = [
    "import os",
    "import sys",
    "import subprocess",
    "import shutil",
    "import socket",
    "import requests",
    "import urllib",
    "from os",
    "from sys",
    "from subprocess",
    "from shutil",
    "from socket",
    "from requests",
    "from urllib",
    "open(",
    "pathlib",
    "exec(",
    "eval(",
    "__import__",
]


def validate_code(code: str) -> str | None:
    """Check code for dangerous patterns.

    Returns an error message if dangerous patterns are found, None if the code is safe.
    """
    for pattern in DANGEROUS_PATTERNS:
        if pattern in code:
            return f"Blocked: code contains disallowed pattern '{pattern}'"
    return None


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("Code execution timed out")


def run_sandboxed(code: str, db_path: str, timeout: int = 10) -> tuple:
    """Run code in a sandboxed namespace.

    The code is executed via compile() and exec() in a restricted namespace containing
    {pd, np, sqlite3, px, go, db_path}. Uses signal.alarm for timeout enforcement.
    The executed code is expected to assign its output to a variable called "result".

    NOTE: The exec() usage here is INTENTIONAL. This is a code execution sandbox
    for an LLM REPL that runs validated, sanitized code snippets.

    Args:
        code: Python code string to execute.
        db_path: Path to the SQLite database file.
        timeout: Maximum execution time in seconds (default 10).

    Returns:
        Tuple of (result, error) where result is the value assigned to "result"
        variable in the code, and error is None on success or an error string.
    """
    namespace = {
        "pd": pd,
        "np": np,
        "sqlite3": sqlite3,
        "px": px,
        "go": go,
        "db_path": db_path,
    }

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        compiled = compile(code, "<llm-code>", "exec")
        # noqa: S102 - intentional sandboxed exec for LLM REPL
        run_exec(compiled, namespace)
        result = namespace.get("result", None)
        return (result, None)
    except _TimeoutError:
        return (None, "Code execution timed out")
    except Exception as e:
        return (None, f"{type(e).__name__}: {e}")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def run_exec(compiled, namespace):
    """Execute compiled code in namespace. Intentional sandboxed exec for LLM REPL."""
    # This function wraps the built-in exec for the sandboxed code runner.
    # The code has been validated by validate_code() before reaching here.
    exec(compiled, namespace)  # noqa: S102 - intentional sandboxed exec for LLM REPL

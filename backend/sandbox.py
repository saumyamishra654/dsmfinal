# sandboxed exec for the chat feature

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
    # quick check for sketchy imports/calls
    for pattern in DANGEROUS_PATTERNS:
        if pattern in code:
            return f"Blocked: code contains disallowed pattern '{pattern}'"
    return None


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("Code execution timed out")


def run_sandboxed(code: str, db_path: str, timeout: int = 10) -> tuple:
    # exec validated code in a restricted namespace, returns (result, error)
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
    # wrapping exec so linters don't yell at the main function
    exec(compiled, namespace)  # noqa: S102

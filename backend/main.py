# fastapi backend for the dashboard

import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chat import handle_chat

app = FastAPI(title="Digital India API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = Path(__file__).parent / "db" / "dsm.db"


def get_con() -> sqlite3.Connection:
    uri = f"file:{DB_PATH}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    return con




class ChatRequest(BaseModel):
    question: str
    api_key: str
    history: Optional[list[dict]] = None


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/states")
def get_states():
    con = get_con()
    try:
        cur = con.execute("SELECT state_id, state_name FROM states ORDER BY state_name")
        rows = [dict(row) for row in cur.fetchall()]
        return rows
    finally:
        con.close()


@app.get("/api/timeseries/{state}")
def get_timeseries(state: str):
    con = get_con()
    try:
        cur = con.execute(
            """
            SELECT td.year, td.month, td.tele_density
            FROM tele_density td
            JOIN states s ON td.state_id = s.state_id
            WHERE s.state_name = ?
            ORDER BY td.year, td.month
            """,
            (state,),
        )
        rows = [dict(row) for row in cur.fetchall()]
        if not rows:
            raise HTTPException(status_code=404, detail=f"No data found for state: {state}")
        return rows
    finally:
        con.close()


@app.get("/api/wireless/{state}")
def get_wireless(state: str):
    con = get_con()
    try:
        cur = con.execute(
            """
            SELECT ww.year, ww.month, ww.wireless_millions, ww.wireline_millions
            FROM wired_wireless ww
            JOIN states s ON ww.state_id = s.state_id
            WHERE s.state_name = ?
            ORDER BY ww.year, ww.month
            """,
            (state,),
        )
        rows = [dict(row) for row in cur.fetchall()]
        if not rows:
            raise HTTPException(status_code=404, detail=f"No data found for state: {state}")
        return rows
    finally:
        con.close()


@app.get("/api/ger/{state}")
def get_ger(state: str):
    con = get_con()
    try:
        cur = con.execute(
            """
            SELECT eg.year, eg.gender, eg.category, eg.ger
            FROM education_ger eg
            JOIN states s ON eg.state_id = s.state_id
            WHERE s.state_name = ?
            ORDER BY eg.year, eg.gender, eg.category
            """,
            (state,),
        )
        rows = [dict(row) for row in cur.fetchall()]
        if not rows:
            raise HTTPException(status_code=404, detail=f"No data found for state: {state}")
        return rows
    finally:
        con.close()


@app.get("/api/national/wireless")
def get_national_wireless():
    con = get_con()
    try:
        cur = con.execute(
            """
            SELECT year, month, SUM(wireless_subscribers) as total_wireless
            FROM telecom_subscriptions
            GROUP BY year, month
            ORDER BY year, month
            """
        )
        rows = [dict(row) for row in cur.fetchall()]
        return rows
    finally:
        con.close()


@app.get("/api/national/transactions")
def get_national_transactions():
    con = get_con()
    try:
        cur = con.execute(
            """
            SELECT year, month, date, digital_txn_crores, bhim_txn_crores, debit_card_crores
            FROM digital_transactions
            ORDER BY year, month
            """
        )
        rows = [dict(row) for row in cur.fetchall()]
        return rows
    finally:
        con.close()


@app.get("/api/hhi")
def get_hhi():
    con = get_con()
    try:
        cur = con.execute(
            """
            WITH state_year_total AS (
                SELECT
                    ts.state as state_name,
                    ts.year,
                    ts.provider,
                    SUM(ts.wireless_subscribers) as provider_total
                FROM telecom_subscriptions ts
                GROUP BY ts.state, ts.year, ts.provider
            ),
            market_total AS (
                SELECT
                    state_name,
                    year,
                    SUM(provider_total) as market_size
                FROM state_year_total
                GROUP BY state_name, year
            ),
            market_shares AS (
                SELECT
                    syt.state_name,
                    syt.year,
                    syt.provider,
                    (syt.provider_total / mt.market_size * 100) as market_share_pct
                FROM state_year_total syt
                JOIN market_total mt
                    ON syt.state_name = mt.state_name AND syt.year = mt.year
                WHERE mt.market_size > 0
            )
            SELECT
                state_name,
                year,
                SUM(market_share_pct * market_share_pct) as hhi
            FROM market_shares
            GROUP BY state_name, year
            ORDER BY state_name, year
            """
        )
        rows = [dict(row) for row in cur.fetchall()]
        return rows
    finally:
        con.close()


@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    result = handle_chat(
        question=request.question,
        api_key=request.api_key,
        history=request.history,
    )
    return result

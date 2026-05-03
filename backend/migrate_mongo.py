# dumps telecom_subscriptions from mongo into sqlite so we don't need mongo running

import sqlite3
from pathlib import Path

from pymongo import MongoClient

DB_PATH = Path(__file__).parent / "db" / "dsm.db"

mongo_client = MongoClient("mongodb://localhost:27017")
mongo_db = mongo_client["dsm"]
collection = mongo_db["telecom_subscriptions"]

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS telecom_subscriptions")
cursor.execute("""
    CREATE TABLE telecom_subscriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        state TEXT,
        telecom_circle TEXT,
        provider TEXT,
        year INTEGER,
        month INTEGER,
        wireless_subscribers REAL,
        wireline_subscribers REAL
    )
""")

docs = collection.find({}, {
    "_id": 0,
    "state": 1,
    "telecom_circle": 1,
    "provider": 1,
    "year": 1,
    "month": 1,
    "wireless_subscribers": 1,
    "wireline_subscribers": 1,
})

rows = []
for doc in docs:
    rows.append((
        doc.get("state"),
        doc.get("telecom_circle"),
        doc.get("provider"),
        doc.get("year"),
        doc.get("month"),
        doc.get("wireless_subscribers"),
        doc.get("wireline_subscribers"),
    ))

cursor.executemany("""
    INSERT INTO telecom_subscriptions
        (state, telecom_circle, provider, year, month, wireless_subscribers, wireline_subscribers)
    VALUES (?, ?, ?, ?, ?, ?, ?)
""", rows)

conn.commit()
conn.close()
mongo_client.close()

print(f"Migration complete. Inserted {len(rows)} rows into telecom_subscriptions.")

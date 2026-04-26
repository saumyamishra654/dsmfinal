"""
Ingests telecom subscription data (provider-level) into MongoDB.

Run directly:
    python src/database_loaders/load_mongo.py
"""

import re
import math
import pandas as pd
from pathlib import Path
from pymongo import MongoClient, ASCENDING

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "datasets" / "telecom subscription data.csv"

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "dsm"
COLLECTION_NAME = "telecom_subscriptions"

# ---------------------------------------------------------------------------
# Date parsing (duplicated from data_cleaning.py to keep loaders standalone)
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _parse_year(raw: str) -> int:
    m = re.search(r"(\d{4})", str(raw))
    if not m:
        raise ValueError(f"Cannot parse year from: {raw!r}")
    return int(m.group(1))


def _parse_month(raw: str) -> int:
    m = re.search(r"([A-Za-z]+)", str(raw))
    if not m:
        raise ValueError(f"Cannot parse month from: {raw!r}")
    name = m.group(1).lower()
    if name not in _MONTH_MAP:
        raise ValueError(f"Unknown month name: {name!r}")
    return _MONTH_MAP[name]


# ---------------------------------------------------------------------------
# Provider name normalization — 47 raw names → ~15 canonical providers
# ---------------------------------------------------------------------------

PROVIDER_MAP = {
    # Bharti/Airtel family
    "Bharti": "Bharti Airtel",
    "Bharti Airtel": "Bharti Airtel",
    "Bharti Airtel (including Tata Tele.)": "Bharti Airtel",
    "Bharti Airtel Ltd.": "Bharti Airtel",
    # Reliance family
    "Reliance": "Reliance Communications",
    "Reliance Com.": "Reliance Communications",
    "Reliance Com. ": "Reliance Communications",
    "Reliance Communications": "Reliance Communications",
    "Reliance Telecom/ Reliance Communication": "Reliance Communications",
    "Reliance Jio": "Reliance Jio",
    # Vodafone family
    "Vodafone": "Vodafone",
    "Vodafone Essar": "Vodafone",
    "Vodafone Idea": "Vodafone Idea",
    # Idea family
    "Idea": "Idea Cellular",
    "Idea/ Spice": "Idea Cellular",
    "Spice": "Idea Cellular",
    # Tata family
    "Tata": "Tata Teleservices",
    "Tata Tele.": "Tata Teleservices",
    "Tata Teleservices": "Tata Teleservices",
    "Teleservices Ltd": "Tata Teleservices",
    # HFCL family
    "HFCL": "HFCL Infotel",
    "HFCL Infotel": "HFCL Infotel",
    "HFCL infotel": "HFCL Infotel",
    "Quadrant": "HFCL Infotel",
    "Quadrant (HFCL)": "HFCL Infotel",
    # Loop family
    "Loop": "Loop Mobile",
    "Loop Mobile(BPL Mobile)": "Loop Mobile",
    "Loop Telecom Pvt. Ltd.": "Loop Mobile",
    "BPL Mobile": "Loop Mobile",
    # Sistema family
    "Sistema": "Sistema Shyam",
    "Sistema Shyam": "Sistema Shyam",
    "Sistema Shyam Teleservices Ltd": "Sistema Shyam",
    "Shyam Telelink": "Sistema Shyam",
    "S-Tel": "Sistema Shyam",
    # Aircel family
    "Aircel": "Aircel",
    "Aircel/Dishnet": "Aircel",
    # Telenor/Uninor family
    "Uninor": "Telenor",
    "Unitech": "Telenor",
    "Telenor": "Telenor",
    "Telewings": "Telenor",
    # Etisalat family
    "Etisalat": "Etisalat",
    "Etisalat/Allianz": "Etisalat",
    # Standalone
    "BSNL": "BSNL",
    "BSNL (Except CDMA)": "BSNL",
    "BSNL (VNOs)": "BSNL VNOs",
    "MTNL": "MTNL",
    "Videocon": "Videocon",
}

# Numeric columns that should be omitted from the document when NaN
_NUMERIC_FIELDS = {
    "Wireless Subscribers (UOM:Number), Scaling Factor:1": "wireless_subscribers",
    "Proportion Of Vlr On A Peak Day (UOM:%(Percentage)), Scaling Factor:1": "vlr_proportion",
    "Wireline Subscribers (UOM:Number), Scaling Factor:1": "wireline_subscribers",
}


# ---------------------------------------------------------------------------
# Core loading logic
# ---------------------------------------------------------------------------

def _row_to_doc(row: pd.Series) -> dict:
    """Convert one CSV row into a MongoDB document, omitting NaN numeric fields."""
    provider_raw = row["Service Provider"]
    provider = PROVIDER_MAP.get(provider_raw, provider_raw)

    doc = {
        "state": row["State"],
        "telecom_circle": row["Telecom Circle Name"],
        "provider": provider,
        "year": _parse_year(row["Year"]),
        "month": _parse_month(row["Month"]),
    }

    for csv_col, mongo_field in _NUMERIC_FIELDS.items():
        val = row[csv_col]
        if pd.notna(val):
            doc[mongo_field] = float(val)

    return doc


def load_to_mongo() -> int:
    """Read the raw CSV, transform rows into documents, and bulk-insert into MongoDB."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {CSV_PATH}. "
            "Ensure the datasets/ directory contains 'telecom subscription data.csv'."
        )

    print(f"Reading {CSV_PATH.name}...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} rows loaded from CSV")

    # Show provider normalization summary
    raw_providers = set(df["Service Provider"].unique())
    unmapped = raw_providers - set(PROVIDER_MAP.keys())
    if unmapped:
        print(f"  WARNING: {len(unmapped)} unmapped providers (kept as-is): {unmapped}")

    print("Converting rows to documents...")
    docs = [_row_to_doc(row) for _, row in df.iterrows()]

    print(f"Connecting to MongoDB ({MONGO_URI})...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    # Drop existing collection for idempotent reload
    db.drop_collection(COLLECTION_NAME)
    print(f"  Dropped existing '{COLLECTION_NAME}' collection (if any)")

    collection = db[COLLECTION_NAME]
    collection.insert_many(docs)
    print(f"  Inserted {len(docs)} documents")

    # Create indexes
    collection.create_index(
        [("state", ASCENDING), ("year", ASCENDING), ("month", ASCENDING)],
        name="idx_state_year_month",
    )
    collection.create_index([("provider", ASCENDING)], name="idx_provider")
    print("  Created indexes: idx_state_year_month, idx_provider")

    # Summary
    n_providers = len(collection.distinct("provider"))
    n_states = len(collection.distinct("state"))
    print(f"\nSummary:")
    print(f"  Documents: {collection.count_documents({})}")
    print(f"  Unique providers (normalized): {n_providers}")
    print(f"  Unique states: {n_states}")

    print("\nSample document:")
    sample = collection.find_one({"provider": "Reliance Jio"})
    if sample:
        sample.pop("_id", None)
        for k, v in sample.items():
            print(f"  {k}: {v}")

    client.close()
    return len(docs)


if __name__ == "__main__":
    n = load_to_mongo()
    print(f"\nDone. {n} documents inserted into {DB_NAME}.{COLLECTION_NAME}")

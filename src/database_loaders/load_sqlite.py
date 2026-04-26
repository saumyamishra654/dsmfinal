import sqlite3
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
CLEANED = ROOT / "cleaned_datasets"
DB_PATH = ROOT / "db" / "dsm.db"

DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS states (
    state_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    state_name TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS tele_density (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    state_id     INTEGER NOT NULL REFERENCES states(state_id),
    year         INTEGER NOT NULL,
    month        INTEGER NOT NULL,
    date         DATE    NOT NULL,
    tele_density REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS wired_wireless (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    state_id          INTEGER NOT NULL REFERENCES states(state_id),
    year              INTEGER NOT NULL,
    month             INTEGER NOT NULL,
    date              DATE    NOT NULL,
    wireline_millions REAL,
    wireless_millions REAL,
    pct_share         REAL
);

CREATE TABLE IF NOT EXISTS education_ger (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    state_id   INTEGER NOT NULL REFERENCES states(state_id),
    year       INTEGER NOT NULL,
    gender     TEXT    NOT NULL,
    category   TEXT    NOT NULL,
    ger        REAL
);

CREATE TABLE IF NOT EXISTS digital_transactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    year                INTEGER NOT NULL,
    month               INTEGER NOT NULL,
    date                DATE    NOT NULL,
    digital_txn_crores  REAL,
    bhim_txn_crores     REAL,
    debit_card_crores   REAL
);

CREATE TABLE IF NOT EXISTS electricity_consumption (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    year            INTEGER NOT NULL,
    sector          TEXT    NOT NULL,
    additional_info TEXT,
    energy_gwh      REAL,
    pct_consumption REAL,
    pct_growth      REAL
);

CREATE INDEX IF NOT EXISTS idx_td_state_date  ON tele_density(state_id, date);
CREATE INDEX IF NOT EXISTS idx_ww_state_date  ON wired_wireless(state_id, date);
CREATE INDEX IF NOT EXISTS idx_ger_state_year ON education_ger(state_id, year);
"""


# inserts a state name if not present and returns its id
def _get_or_create_state(cur, name):
    cur.execute("INSERT OR IGNORE INTO states(state_name) VALUES (?)", (name,))
    cur.execute("SELECT state_id FROM states WHERE state_name = ?", (name,))
    return cur.fetchone()[0]


# ensures all state names exist in the states table and returns name->id dict
def _state_id_map(cur, names):
    mapping = {}
    for name in names.unique():
        mapping[name] = _get_or_create_state(cur, name)
    return mapping


# loads tele_density rows from pickle into SQLite
def load_tele_density(cur):
    df = pd.read_pickle(CLEANED / "tele_density.pkl")
    sid = _state_id_map(cur, df["state_name"])
    rows = [
        (sid[r.state_name], r.year, r.month, str(r.date.date()), r.tele_density)
        for r in df.itertuples(index=False)
    ]
    cur.executemany(
        "INSERT INTO tele_density(state_id, year, month, date, tele_density) VALUES (?,?,?,?,?)",
        rows,
    )
    return len(rows)


# loads wired_wireless rows from pickle into SQLite
def load_wired_wireless(cur):
    df = pd.read_pickle(CLEANED / "wired_wireless.pkl")
    sid = _state_id_map(cur, df["state_name"])
    rows = [
        (sid[r.state_name], r.year, r.month, str(r.date.date()),
         r.wireline_millions, r.wireless_millions, r.pct_share)
        for r in df.itertuples(index=False)
    ]
    cur.executemany(
        """INSERT INTO wired_wireless
           (state_id, year, month, date, wireline_millions, wireless_millions, pct_share)
           VALUES (?,?,?,?,?,?,?)""",
        rows,
    )
    return len(rows)


# loads education_ger rows from pickle into SQLite
def load_education_ger(cur):
    df = pd.read_pickle(CLEANED / "education_ger.pkl")
    sid = _state_id_map(cur, df["state_name"])
    rows = [
        (sid[r.state_name], r.year, r.gender, r.category,
         None if pd.isna(r.ger) else r.ger)
        for r in df.itertuples(index=False)
    ]
    cur.executemany(
        "INSERT INTO education_ger(state_id, year, gender, category, ger) VALUES (?,?,?,?,?)",
        rows,
    )
    return len(rows)


# loads digital_transactions rows from pickle into SQLite
def load_digital_transactions(cur):
    df = pd.read_pickle(CLEANED / "digital_transactions.pkl")
    rows = [
        (r.year, r.month, str(r.date.date()),
         r.digital_txn_crores, r.bhim_txn_crores, r.debit_card_crores)
        for r in df.itertuples(index=False)
    ]
    cur.executemany(
        """INSERT INTO digital_transactions
           (year, month, date, digital_txn_crores, bhim_txn_crores, debit_card_crores)
           VALUES (?,?,?,?,?,?)""",
        rows,
    )
    return len(rows)


# loads electricity_consumption rows from pickle into SQLite
def load_electricity(cur):
    df = pd.read_pickle(CLEANED / "electricity.pkl")
    rows = [
        (r.year, r.sector, r.additional_info, r.energy_gwh,
         r.pct_consumption, None if pd.isna(r.pct_growth) else r.pct_growth)
        for r in df.itertuples(index=False)
    ]
    cur.executemany(
        """INSERT INTO electricity_consumption
           (year, sector, additional_info, energy_gwh, pct_consumption, pct_growth)
           VALUES (?,?,?,?,?,?)""",
        rows,
    )
    return len(rows)


if __name__ == "__main__":
    DB_PATH.parent.mkdir(exist_ok=True)

    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Removed existing {DB_PATH.name}")

    con = sqlite3.connect(DB_PATH)
    con.executescript(DDL)

    loaders = [
        ("tele_density",        load_tele_density),
        ("wired_wireless",      load_wired_wireless),
        ("education_ger",       load_education_ger),
        ("digital_transactions", load_digital_transactions),
        ("electricity_consumption", load_electricity),
    ]

    with con:
        cur = con.cursor()
        for table, fn in loaders:
            n = fn(cur)
            print(f"  {table}: {n} rows inserted")

    print("\nRow counts in DB:")
    for table, _ in loaders:
        (count,) = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        print(f"  {table}: {count}")

    states_count = con.execute("SELECT COUNT(*) FROM states").fetchone()[0]
    print(f"  states: {states_count}")

    con.close()
    print(f"\nDatabase written to {DB_PATH}")

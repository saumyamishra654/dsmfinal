"""
Cleans all raw CSVs and writes typed DataFrames to cleaned_datasets/*.pkl

Run directly:
    python src/data/clean.py
"""

import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
CLEANED = ROOT / "cleaned_datasets"

# ---------------------------------------------------------------------------
# Date parsing
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


def _make_date(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}-01"


def _fy_to_cy_year(fy_year: int, month: int) -> int:
    """FY label → calendar year. Months Apr–Dec belong to (fy_year - 1)."""
    return fy_year - 1 if month >= 4 else fy_year


# ---------------------------------------------------------------------------
# Circle → State mapping
# ---------------------------------------------------------------------------

CIRCLE_TO_STATE: dict[str, str] = {
    "Andhra Pradesh":       "Andhra Pradesh",
    "Assam":                "Assam",
    "Bihar":                "Bihar",
    "Delhi":                "Delhi",
    "Gujarat":              "Gujarat",
    "Haryana":              "Haryana",
    "Himachal Pradesh":     "Himachal Pradesh",
    "Jammu & Kashmir":      "Jammu And Kashmir",
    "Karnataka":            "Karnataka",
    "Kerala":               "Kerala",
    "Madhya Pradesh":       "Madhya Pradesh",
    "Maharashtra":          "Maharashtra",
    "Odisha":               "Odisha",
    "Punjab":               "Punjab",
    "Rajasthan":            "Rajasthan",
    "Tamil Nadu":           "Tamil Nadu",
    "Uttar Pradesh":        "Uttar Pradesh",
    "Uttar Pradesh (East)": "Uttar Pradesh",
    "Uttar Pradesh (West)": "Uttar Pradesh",
    "West Bengal":          "West Bengal",
}

PANEL_STATES: list[str] = sorted(set(CIRCLE_TO_STATE.values()))

# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------

def _clean_tele_density() -> pd.DataFrame:
    df = pd.read_csv(DATASETS / "area-wise tele density.csv")
    df.columns = ["country", "year_raw", "month_raw", "circle", "tele_density"]

    df["month"] = df["month_raw"].apply(_parse_month)
    df["year"]  = df["year_raw"].apply(_parse_year).combine(
        df["month"], _fy_to_cy_year
    )
    df["date"]       = pd.to_datetime(df.apply(lambda r: _make_date(r["year"], r["month"]), axis=1))
    df["state_name"] = df["circle"].map(CIRCLE_TO_STATE)

    df = df[df["state_name"].notna()].copy()
    return df[["state_name", "year", "month", "date", "tele_density"]].reset_index(drop=True)


def _clean_wired_wireless() -> pd.DataFrame:
    df = pd.read_csv(DATASETS / "wired, wireless telephone.csv")
    df.columns = ["country", "year_raw", "month_raw", "circle",
                  "wireline_millions", "wireless_millions", "pct_share"]

    df["month"]      = df["month_raw"].apply(_parse_month)
    df["year"]       = df["year_raw"].apply(_parse_year)   # CY format, no FY shift
    df["date"]       = pd.to_datetime(df.apply(lambda r: _make_date(r["year"], r["month"]), axis=1))
    df["state_name"] = df["circle"].map(CIRCLE_TO_STATE)

    df = df[df["state_name"].notna()].copy()

    # Aggregate UP East + UP West into one row per month
    up = df["state_name"] == "Uttar Pradesh"
    up_agg = (
        df[up]
        .groupby(["state_name", "year", "month", "date"], as_index=False)
        .agg(wireline_millions=("wireline_millions", "sum"),
             wireless_millions=("wireless_millions", "sum"))
    )
    total = up_agg["wireline_millions"] + up_agg["wireless_millions"]
    up_agg["pct_share"] = (up_agg["wireless_millions"] / total * 100).round(2)

    out = pd.concat([df[~up], up_agg], ignore_index=True)
    return out[["state_name", "year", "month", "date",
                "wireline_millions", "wireless_millions", "pct_share"]].reset_index(drop=True)


def _clean_education_ger() -> pd.DataFrame:
    df = pd.read_csv(DATASETS / "education-enrolment.csv")
    df.columns = ["country", "state_name", "year_raw", "gender", "category", "ger"]

    df["year"] = df["year_raw"].apply(_parse_year)
    df["ger"]  = pd.to_numeric(df["ger"], errors="coerce")

    return df[["state_name", "year", "gender", "category", "ger"]].reset_index(drop=True)


def _clean_digital_transactions() -> pd.DataFrame:
    df = pd.read_csv(DATASETS / "digital transactions.csv")
    df.columns = ["country", "year_raw", "month_raw", "ministry", "project",
                  "digital_txn_crores", "bhim_txn_crores", "debit_card_crores"]

    df["month"] = df["month_raw"].apply(_parse_month)
    df["year"]  = df["year_raw"].apply(_parse_year)   # CY format
    df["date"]  = pd.to_datetime(df.apply(lambda r: _make_date(r["year"], r["month"]), axis=1))

    return df[["year", "month", "date",
               "digital_txn_crores", "bhim_txn_crores", "debit_card_crores"]].reset_index(drop=True)


def _clean_electricity() -> pd.DataFrame:
    df = pd.read_csv(DATASETS / "sector-wise electricity consumption.csv")
    df.columns = ["country", "year_raw", "sector", "additional_info",
                  "energy_gwh", "pct_consumption", "pct_growth"]

    df["year"]       = df["year_raw"].apply(_parse_year)
    df["pct_growth"] = pd.to_numeric(df["pct_growth"], errors="coerce")

    return df[["year", "sector", "additional_info",
               "energy_gwh", "pct_consumption", "pct_growth"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main — write all cleaned datasets to cleaned_datasets/
# ---------------------------------------------------------------------------

CLEANERS = {
    "tele_density":        _clean_tele_density,
    "wired_wireless":      _clean_wired_wireless,
    "education_ger":       _clean_education_ger,
    "digital_transactions": _clean_digital_transactions,
    "electricity":         _clean_electricity,
}

if __name__ == "__main__":
    CLEANED.mkdir(exist_ok=True)

    for name, fn in CLEANERS.items():
        print(f"Cleaning {name}...", end=" ")
        df = fn()
        out = CLEANED / f"{name}.pkl"
        df.to_pickle(out)
        print(f"{len(df)} rows → {out.name}")

    print("\nDone. Cleaned datasets written to cleaned_datasets/")

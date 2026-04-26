import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
CLEANED = ROOT / "cleaned_datasets"

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


# extracts the 4-digit year from a TRAI year string
def _parse_year(raw):
    m = re.search(r"(\d{4})", str(raw))
    if not m:
        raise ValueError(f"Cannot parse year from: {raw!r}")
    return int(m.group(1))


# extracts the month name from a TRAI month string and maps to int
def _parse_month(raw):
    m = re.search(r"([A-Za-z]+)", str(raw))
    if not m:
        raise ValueError(f"Cannot parse month from: {raw!r}")
    name = m.group(1).lower()
    if name not in _MONTH_MAP:
        raise ValueError(f"Unknown month name: {name!r}")
    return _MONTH_MAP[name]


# builds the YYYY-MM-01 date string for SQL DATE columns
def _make_date(year, month):
    return f"{year:04d}-{month:02d}-01"


# converts a Financial Year label to calendar year (Apr-Dec belong to fy_year - 1)
def _fy_to_cy_year(fy_year, month):
    return fy_year - 1 if month >= 4 else fy_year


CIRCLE_TO_STATE = {
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

PANEL_STATES = sorted(set(CIRCLE_TO_STATE.values()))


# cleans the area-wise tele-density CSV into a typed DataFrame
def _clean_tele_density():
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


# cleans the wired/wireless CSV and aggregates UP East + UP West
def _clean_wired_wireless():
    df = pd.read_csv(DATASETS / "wired, wireless telephone.csv")
    df.columns = ["country", "year_raw", "month_raw", "circle",
                  "wireline_millions", "wireless_millions", "pct_share"]

    df["month"]      = df["month_raw"].apply(_parse_month)
    df["year"]       = df["year_raw"].apply(_parse_year)
    df["date"]       = pd.to_datetime(df.apply(lambda r: _make_date(r["year"], r["month"]), axis=1))
    df["state_name"] = df["circle"].map(CIRCLE_TO_STATE)

    df = df[df["state_name"].notna()].copy()

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


# cleans the education GER CSV
def _clean_education_ger():
    df = pd.read_csv(DATASETS / "education-enrolment.csv")
    df.columns = ["country", "state_name", "year_raw", "gender", "category", "ger"]

    df["year"] = df["year_raw"].apply(_parse_year)
    df["ger"]  = pd.to_numeric(df["ger"], errors="coerce")

    return df[["state_name", "year", "gender", "category", "ger"]].reset_index(drop=True)


# cleans the digital transactions CSV
def _clean_digital_transactions():
    df = pd.read_csv(DATASETS / "digital transactions.csv")
    df.columns = ["country", "year_raw", "month_raw", "ministry", "project",
                  "digital_txn_crores", "bhim_txn_crores", "debit_card_crores"]

    df["month"] = df["month_raw"].apply(_parse_month)
    df["year"]  = df["year_raw"].apply(_parse_year)
    df["date"]  = pd.to_datetime(df.apply(lambda r: _make_date(r["year"], r["month"]), axis=1))

    return df[["year", "month", "date",
               "digital_txn_crores", "bhim_txn_crores", "debit_card_crores"]].reset_index(drop=True)


# cleans the sector-wise electricity consumption CSV
def _clean_electricity():
    df = pd.read_csv(DATASETS / "sector-wise electricity consumption.csv")
    df.columns = ["country", "year_raw", "sector", "additional_info",
                  "energy_gwh", "pct_consumption", "pct_growth"]

    df["year"]       = df["year_raw"].apply(_parse_year)
    df["pct_growth"] = pd.to_numeric(df["pct_growth"], errors="coerce")

    return df[["year", "sector", "additional_info",
               "energy_gwh", "pct_consumption", "pct_growth"]].reset_index(drop=True)


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

"""
Objective 5: Electricity consumption as corroboration proxy

Data sources: SQLite electricity_consumption (national, 1970-2023)
              + MongoDB national wireless subscribers (2008-2021)
              + SQLite digital transactions (2017-2022)

Run directly:
    python src/analysis/obj5_electricity.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.analysis.obj1_wireless_growth import get_national_wireless_ts

DB_PATH = ROOT / "db" / "sqlite" / "dsm.db"
FIGURES = ROOT / "outputs" / "figures"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_electricity_domestic_commercial() -> pd.DataFrame:
    """Annual Domestic+Commercial electricity consumption from SQLite."""
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT year, SUM(energy_gwh) as energy_gwh
        FROM electricity_consumption
        WHERE sector IN ('Domestic', 'Commercial')
        GROUP BY year
        ORDER BY year
    """, con)
    con.close()
    return df


def get_national_wireless_annual() -> pd.DataFrame:
    """Annual national wireless subscribers from MongoDB (2008-2021)."""
    ts = get_national_wireless_ts()
    annual = ts.groupby("year")["total_wireless"].mean().reset_index()
    annual.columns = ["year", "wireless_avg"]
    return annual


def get_digital_txn_annual() -> pd.DataFrame:
    """Annual average digital transaction volume from SQLite."""
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT year, AVG(digital_txn_crores) as avg_txn
        FROM digital_transactions
        GROUP BY year
        ORDER BY year
    """, con)
    con.close()
    return df


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalization to [0, 1]."""
    smin, smax = s.min(), s.max()
    if smax == smin:
        return pd.Series(0.5, index=s.index)
    return (s - smin) / (smax - smin)


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int = 5) -> pd.DataFrame:
    """Compute Pearson correlation at lags from -max_lag to +max_lag.

    Positive lag means x leads y (x at time t correlates with y at time t+lag).
    """
    results = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x_shifted = x[-lag:]
            y_shifted = y[:lag]
        elif lag > 0:
            x_shifted = x[:-lag]
            y_shifted = y[lag:]
        else:
            x_shifted = x
            y_shifted = y

        if len(x_shifted) < 4:
            continue

        corr, pval = stats.pearsonr(x_shifted, y_shifted)
        results.append({"lag": lag, "correlation": corr, "p_value": pval})

    return pd.DataFrame(results)


def growth_acceleration_test(energy_df: pd.DataFrame,
                             break_year: int = 2010) -> tuple:
    """Test if domestic+commercial electricity growth accelerated after break_year.

    Uses only post-2000 data where year spacing is annual.
    Returns (pre_mean, post_mean, t_stat, p_value).
    """
    df = energy_df[energy_df["year"] >= 2000].copy()
    df = df.sort_values("year")
    df["growth"] = df["energy_gwh"].pct_change()
    df = df.dropna(subset=["growth"])

    pre = df[df["year"] <= break_year]["growth"]
    post = df[df["year"] > break_year]["growth"]

    if len(pre) < 2 or len(post) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    t_stat, p_val = stats.ttest_ind(post, pre, equal_var=False)
    return pre.mean(), post.mean(), t_stat, p_val


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_normalized_overlay(elec_df: pd.DataFrame, wireless_df: pd.DataFrame,
                            txn_df: pd.DataFrame) -> None:
    """Overlaid normalized time series of electricity, wireless, and digital txn."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Electricity (full range)
    ax.plot(elec_df["year"], normalize_series(elec_df["energy_gwh"]),
            linewidth=2, color="#1f77b4", label="Domestic+Commercial Electricity")

    # Wireless (2008-2021)
    ax.plot(wireless_df["year"], normalize_series(wireless_df["wireless_avg"]),
            linewidth=2, color="#ff7f0e", label="Wireless Subscribers")

    # Digital transactions (2017-2022)
    ax.plot(txn_df["year"], normalize_series(txn_df["avg_txn"]),
            linewidth=2, color="#2ca02c", label="Digital Transactions")

    # Annotations
    ax.axvline(2010, color="gray", linestyle="--", alpha=0.5)
    ax.annotate("Digital\nacceleration", xy=(2010, 0.95), fontsize=8,
                color="gray", ha="center")
    ax.axvline(2016, color="red", linestyle="--", alpha=0.5)
    ax.annotate("Jio entry", xy=(2016, 0.85), fontsize=8,
                color="red", ha="center")

    ax.set_xlabel("Year")
    ax.set_ylabel("Normalized Value (0–1)")
    ax.set_title("Synchronized Growth: Electricity, Wireless Subscribers, Digital Transactions")
    ax.legend(loc="upper left")
    ax.set_xlim(1990, 2023)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "obj5_normalized_overlay.png", dpi=150)
    plt.close(fig)
    print(f"  Saved obj5_normalized_overlay.png")


def plot_ccf(ccf_df: pd.DataFrame, n_obs: int) -> None:
    """Bar chart of cross-correlation at different lags."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2ca02c" if p < 0.05 else "#cccccc" for p in ccf_df["p_value"]]
    ax.bar(ccf_df["lag"], ccf_df["correlation"], color=colors,
           edgecolor="black", linewidth=0.5)

    # Significance bands
    sig_band = 2.0 / np.sqrt(n_obs)
    ax.axhline(sig_band, color="red", linestyle=":", alpha=0.5)
    ax.axhline(-sig_band, color="red", linestyle=":", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.5)

    ax.set_xlabel("Lag (years) — positive = electricity leads wireless")
    ax.set_ylabel("Pearson Correlation")
    ax.set_title("Cross-Correlation: Electricity Growth vs Wireless Subscriber Growth")
    ax.set_xticks(ccf_df["lag"])
    ax.grid(True, axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#2ca02c", label="Significant (p < 0.05)"),
        Patch(facecolor="#cccccc", label="Not significant"),
    ])

    fig.tight_layout()
    fig.savefig(FIGURES / "obj5_ccf.png", dpi=150)
    plt.close(fig)
    print(f"  Saved obj5_ccf.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    # 1. Load data
    print("1. Loading data...")
    elec = get_electricity_domestic_commercial()
    print(f"   Electricity: {len(elec)} years ({elec['year'].min()}–{elec['year'].max()})")

    wireless = get_national_wireless_annual()
    print(f"   Wireless: {len(wireless)} years ({wireless['year'].min()}–{wireless['year'].max()})")

    txn = get_digital_txn_annual()
    print(f"   Digital txn: {len(txn)} years ({txn['year'].min()}–{txn['year'].max()})")

    # 2. Normalized overlay plot
    print("\n2. Generating normalized overlay...")
    plot_normalized_overlay(elec, wireless, txn)

    # 3. Cross-correlation (electricity growth vs wireless growth)
    print("\n3. Computing cross-correlation...")
    # Use only post-2008 annual data where both series exist
    elec_post = elec[elec["year"] >= 2008].sort_values("year").reset_index(drop=True)
    wire_post = wireless.sort_values("year").reset_index(drop=True)

    # Merge on year
    merged = pd.merge(elec_post, wire_post, on="year")
    merged["elec_growth"] = merged["energy_gwh"].pct_change()
    merged["wireless_growth"] = merged["wireless_avg"].pct_change()
    merged = merged.dropna()

    if len(merged) >= 6:
        ccf = cross_correlation(
            merged["elec_growth"].values,
            merged["wireless_growth"].values,
            max_lag=3,  # limited by ~13 overlapping years
        )
        print(f"   {len(merged)} overlapping annual growth observations")
        print("   Cross-correlation results:")
        for _, row in ccf.iterrows():
            sig = "*" if row["p_value"] < 0.05 else ""
            print(f"     Lag {int(row['lag']):+d}: r={row['correlation']:.3f}, p={row['p_value']:.3f} {sig}")

        # Find peak
        peak = ccf.loc[ccf["correlation"].abs().idxmax()]
        direction = "electricity leads" if peak["lag"] > 0 else "wireless leads" if peak["lag"] < 0 else "contemporaneous"
        print(f"   Peak correlation at lag {int(peak['lag'])}: r={peak['correlation']:.3f} ({direction})")

        plot_ccf(ccf, len(merged))
    else:
        print(f"   Only {len(merged)} overlapping points — insufficient for CCF")

    # 4. Growth acceleration test
    print("\n4. Testing growth acceleration (Welch's t-test)...")
    pre_mean, post_mean, t_stat, p_val = growth_acceleration_test(elec, break_year=2010)
    print(f"   Pre-2010 avg annual growth:  {pre_mean * 100:.2f}%")
    print(f"   Post-2010 avg annual growth: {post_mean * 100:.2f}%")
    print(f"   t-statistic: {t_stat:.3f}")
    print(f"   p-value:     {p_val:.4f}")
    if p_val < 0.05:
        if post_mean > pre_mean:
            print("   Result: Post-2010 growth is significantly HIGHER")
        else:
            print("   Result: Post-2010 growth is significantly LOWER")
    else:
        print("   Result: No significant difference in growth rates")

    print("\nObjective 5 complete.")


if __name__ == "__main__":
    main()

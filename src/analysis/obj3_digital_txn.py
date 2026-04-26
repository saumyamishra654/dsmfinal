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
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

DB_PATH = ROOT / "db" / "sqlite" / "dsm.db"
FIGURES = ROOT / "outputs" / "figures"


# loads digital transactions from SQLite
def get_digital_transactions():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT year, month, date, digital_txn_crores,
               bhim_txn_crores, debit_card_crores
        FROM digital_transactions
        ORDER BY date
    """, con)
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


# national wireless subscribers from SQLite wired_wireless (2017-2023)
def get_wireless_for_granger():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT year, month, date,
               SUM(wireless_millions) as national_wireless
        FROM wired_wireless
        GROUP BY year, month
        ORDER BY date
    """, con)
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


# computes UPI/BHIM, debit card, and other payment shares
def compute_payment_shares(df):
    df = df.copy()
    df["upi_share"] = df["bhim_txn_crores"] / df["digital_txn_crores"]
    df["debit_share"] = df["debit_card_crores"] / df["digital_txn_crores"]
    df["other_share"] = 1 - df["upi_share"] - df["debit_share"]
    return df


# runs STL decomposition on the digital transaction time series
def run_stl(df):
    ts = df.set_index("date")["digital_txn_crores"].copy()
    ts = ts.asfreq("MS")
    ts = ts.interpolate()
    stl = STL(ts, period=12, seasonal=13, robust=True)
    return stl.fit()


# tests if wireless subscriber growth Granger-causes digital txn growth
def granger_causality(wireless_df, txn_df, max_lag=4):
    merged = pd.merge(
        wireless_df[["year", "month", "national_wireless"]],
        txn_df[["year", "month", "digital_txn_crores"]],
        on=["year", "month"],
    ).sort_values(["year", "month"]).reset_index(drop=True)

    merged["wireless_growth"] = merged["national_wireless"].pct_change()
    merged["txn_growth"] = merged["digital_txn_crores"].pct_change()
    merged = merged.dropna(subset=["wireless_growth", "txn_growth"])

    for col in ["wireless_growth", "txn_growth"]:
        adf_stat, adf_p, *_ = adfuller(merged[col].dropna())
        status = "stationary" if adf_p < 0.05 else "NON-STATIONARY"
        print(f"   ADF test on {col}: stat={adf_stat:.3f}, p={adf_p:.4f} ({status})")

    data = merged[["txn_growth", "wireless_growth"]].values

    results = {}
    try:
        gc = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        for lag, tests in gc.items():
            f_stat = tests[0]["ssr_ftest"][0]
            p_val = tests[0]["ssr_ftest"][1]
            results[lag] = (f_stat, p_val)
    except Exception as e:
        print(f"   Granger test failed: {e}")

    return results


# 4-panel STL decomposition plot with COVID annotation
def plot_stl_decomposition(stl_result, figures_dir):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    covid_date = pd.Timestamp("2020-03-01")

    components = [
        ("Original", stl_result.observed),
        ("Trend", stl_result.trend),
        ("Seasonal", stl_result.seasonal),
        ("Residual", stl_result.resid),
    ]

    for ax, (title, data) in zip(axes, components):
        ax.plot(data.index, data.values, linewidth=1.5)
        if title in ("Original", "Trend"):
            ax.axvline(covid_date, color="red", linestyle="--", alpha=0.6)
            ax.annotate("COVID", xy=(covid_date, ax.get_ylim()[1] * 0.8),
                        fontsize=8, color="red")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("STL Decomposition — Digital Transaction Volume (Crores INR)")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(figures_dir / "obj3_stl_decomposition.png", dpi=150)
    plt.close(fig)
    print(f"  Saved obj3_stl_decomposition.png")


# stacked area chart of payment method shares
def plot_payment_shares(df, figures_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(
        df["date"],
        df["upi_share"] * 100,
        df["debit_share"] * 100,
        df["other_share"] * 100,
        labels=["UPI/BHIM", "Debit Card", "Other Digital"],
        colors=["#2ca02c", "#ff7f0e", "#9467bd"],
        alpha=0.85,
    )
    covid_date = pd.Timestamp("2020-03-01")
    ax.axvline(covid_date, color="red", linestyle="--", alpha=0.6)
    ax.annotate("COVID", xy=(covid_date, 95), fontsize=9, color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Share of Total Digital Transactions (%)")
    ax.set_title("Payment Method Composition — UPI vs Debit Card vs Other")
    ax.set_ylim(0, 100)
    ax.legend(loc="center left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "obj3_payment_shares.png", dpi=150)
    plt.close(fig)
    print(f"  Saved obj3_payment_shares.png")


# bar chart of Granger causality F-statistics by lag
def plot_granger_results(results, figures_dir):
    if not results:
        print("  Skipping Granger plot (no results)")
        return

    lags = sorted(results.keys())
    f_stats = [results[lag][0] for lag in lags]
    p_vals = [results[lag][1] for lag in lags]
    colors = ["#2ca02c" if p < 0.05 else "#cccccc" for p in p_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(lags, f_stats, color=colors, edgecolor="black", linewidth=0.5)

    for lag, f, p in zip(lags, f_stats, p_vals):
        ax.annotate(f"p={p:.3f}", xy=(lag, f), ha="center", va="bottom",
                    fontsize=8, color="black")

    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("F-statistic")
    ax.set_title("Granger Causality: Wireless Growth → Digital Txn Growth")
    ax.set_xticks(lags)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#2ca02c", label="Significant (p < 0.05)"),
        Patch(facecolor="#cccccc", label="Not significant"),
    ])
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "obj3_granger_causality.png", dpi=150)
    plt.close(fig)
    print(f"  Saved obj3_granger_causality.png")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    print("1. Loading digital transactions from SQLite...")
    txn = get_digital_transactions()
    print(f"   {len(txn)} rows ({txn['date'].min().date()} to {txn['date'].max().date()})")

    print("\n2. Computing payment shares...")
    txn = compute_payment_shares(txn)
    first_upi = txn.iloc[0]["upi_share"] * 100
    last_upi = txn.iloc[-1]["upi_share"] * 100
    print(f"   UPI share: {first_upi:.1f}% (earliest) → {last_upi:.1f}% (latest)")

    first_debit = txn.iloc[0]["debit_share"] * 100
    last_debit = txn.iloc[-1]["debit_share"] * 100
    print(f"   Debit share: {first_debit:.1f}% (earliest) → {last_debit:.1f}% (latest)")

    print("\n3. Running STL decomposition...")
    stl_result = run_stl(txn)
    trend = stl_result.trend
    trend_direction = "increasing" if trend.iloc[-1] > trend.iloc[0] else "decreasing"
    print(f"   Trend direction: {trend_direction}")
    print(f"   Trend range: {trend.min():.1f} → {trend.max():.1f} crores")

    print("\n4. Testing Granger causality (wireless → digital txn)...")
    wireless = get_wireless_for_granger()
    print(f"   Wireless data: {len(wireless)} months ({wireless['date'].min().date()} to {wireless['date'].max().date()})")
    gc_results = granger_causality(wireless, txn, max_lag=4)

    if gc_results:
        print(f"   Results:")
        for lag in sorted(gc_results):
            f_stat, p_val = gc_results[lag]
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            print(f"     Lag {lag}: F={f_stat:.3f}, p={p_val:.4f} {sig}")

    print("\n5. Generating visualizations...")
    plot_stl_decomposition(stl_result, FIGURES)
    plot_payment_shares(txn, FIGURES)
    plot_granger_results(gc_results, FIGURES)

    print("\n--- REPORT NOTE ---")
    print("Digital transactions data is national-level only — cannot replicate")
    print("this analysis as a state panel. This is a known data limitation.")

    print("\nObjective 3 complete.")


if __name__ == "__main__":
    main()

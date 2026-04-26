"""
Objective 1: Wireless subscriber growth + Jio structural break detection (2008-2021)

Data source: MongoDB telecom_subscriptions collection

Run directly:
    python src/analysis/obj1_wireless_growth.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pymongo import MongoClient
from scipy import stats
import ruptures as rpt

FIGURES = ROOT / "outputs" / "figures"
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "dsm"


# ---------------------------------------------------------------------------
# Shared utility — imported by obj3 and obj5
# ---------------------------------------------------------------------------

def get_national_wireless_ts(db=None) -> pd.DataFrame:
    """National wireless subscriber time series from MongoDB (2008-2021).

    Returns DataFrame with columns [year, month, date, total_wireless].
    """
    if db is None:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]

    pipeline = [
        {"$match": {"wireless_subscribers": {"$exists": True}}},
        {"$group": {
            "_id": {"year": "$year", "month": "$month"},
            "total_wireless": {"$sum": "$wireless_subscribers"},
        }},
        {"$sort": {"_id.year": 1, "_id.month": 1}},
        {"$project": {
            "_id": 0,
            "year": "$_id.year",
            "month": "$_id.month",
            "total_wireless": 1,
        }},
    ]
    docs = list(db.telecom_subscriptions.aggregate(pipeline))
    df = pd.DataFrame(docs)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Structural break detection
# ---------------------------------------------------------------------------

def detect_structural_breaks(series: np.ndarray, n_bkps: int = 2) -> list:
    """Find structural breakpoints using Binseg (L2 cost model).

    Returns list of breakpoint indices (exclusive end of each segment).
    With n_bkps=2, expects to find both the ~2011 mobile explosion break
    and the ~2016 Jio entry break.
    """
    algo = rpt.Binseg(model="l2", min_size=12).fit(series)
    result = algo.predict(n_bkps=n_bkps)
    return result[:-1]  # drop the last element (always == len(series))


def chow_test(y: np.ndarray, break_idx: int) -> tuple:
    """Chow test for structural break at break_idx.

    Tests H0: same linear trend across both sub-periods.
    Returns (F_statistic, p_value).
    """
    n = len(y)
    x = np.arange(n)
    k = 2  # intercept + slope

    # Pooled regression
    c_pool = np.polyfit(x, y, 1)
    rss_pool = np.sum((y - np.polyval(c_pool, x)) ** 2)

    # Sub-sample 1
    x1, y1 = x[:break_idx], y[:break_idx]
    c1 = np.polyfit(x1, y1, 1)
    rss1 = np.sum((y1 - np.polyval(c1, x1)) ** 2)

    # Sub-sample 2
    x2, y2 = x[break_idx:], y[break_idx:]
    c2 = np.polyfit(x2, y2, 1)
    rss2 = np.sum((y2 - np.polyval(c2, x2)) ** 2)

    df_num = k
    df_den = n - 2 * k
    f_stat = ((rss_pool - rss1 - rss2) / df_num) / ((rss1 + rss2) / df_den)
    p_value = stats.f.sf(f_stat, df_num, df_den)
    return f_stat, p_value


# ---------------------------------------------------------------------------
# CAGR
# ---------------------------------------------------------------------------

def compute_cagr(start_val: float, end_val: float, years: float) -> float:
    """Compound annual growth rate."""
    if start_val <= 0 or years <= 0:
        return float("nan")
    return (end_val / start_val) ** (1 / years) - 1


# ---------------------------------------------------------------------------
# State-level growth analysis
# ---------------------------------------------------------------------------

def get_state_growth_rates(db, break_year: int, break_month: int) -> pd.DataFrame:
    """Pre/post-break CAGR per state from MongoDB."""
    pipeline = [
        {"$match": {"wireless_subscribers": {"$exists": True}}},
        {"$group": {
            "_id": {"state": "$state", "year": "$year", "month": "$month"},
            "total_wireless": {"$sum": "$wireless_subscribers"},
        }},
        {"$sort": {"_id.year": 1, "_id.month": 1}},
    ]
    docs = list(db.telecom_subscriptions.aggregate(pipeline))
    df = pd.DataFrame([{
        "state": d["_id"]["state"],
        "year": d["_id"]["year"],
        "month": d["_id"]["month"],
        "total_wireless": d["total_wireless"],
    } for d in docs])
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))

    break_date = pd.Timestamp(year=break_year, month=break_month, day=1)
    results = []

    for state, sdf in df.groupby("state"):
        sdf = sdf.sort_values("date")
        pre = sdf[sdf["date"] < break_date]
        post = sdf[sdf["date"] >= break_date]

        if len(pre) < 2 or len(post) < 2:
            continue

        pre_years = (pre["date"].iloc[-1] - pre["date"].iloc[0]).days / 365.25
        post_years = (post["date"].iloc[-1] - post["date"].iloc[0]).days / 365.25

        pre_cagr = compute_cagr(pre["total_wireless"].iloc[0],
                                pre["total_wireless"].iloc[-1], pre_years)
        post_cagr = compute_cagr(post["total_wireless"].iloc[0],
                                 post["total_wireless"].iloc[-1], post_years)

        results.append({
            "state": state,
            "pre_cagr": pre_cagr,
            "post_cagr": post_cagr,
            "acceleration": post_cagr - pre_cagr,
        })

    return pd.DataFrame(results).sort_values("acceleration", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# HHI (Herfindahl-Hirschman Index)
# ---------------------------------------------------------------------------

def compute_hhi(db) -> pd.DataFrame:
    """HHI per state per year from MongoDB."""
    pipeline = [
        {"$match": {"wireless_subscribers": {"$exists": True}}},
        {"$group": {
            "_id": {"state": "$state", "year": "$year", "provider": "$provider"},
            "provider_total": {"$sum": "$wireless_subscribers"},
        }},
        {"$group": {
            "_id": {"state": "$_id.state", "year": "$_id.year"},
            "providers": {"$push": {
                "provider": "$_id.provider",
                "total": "$provider_total",
            }},
            "state_total": {"$sum": "$provider_total"},
        }},
        {"$sort": {"_id.year": 1}},
    ]
    docs = list(db.telecom_subscriptions.aggregate(pipeline))

    rows = []
    for doc in docs:
        state_total = doc["state_total"]
        if state_total <= 0:
            continue
        hhi = sum((p["total"] / state_total) ** 2 for p in doc["providers"]) * 10000
        rows.append({
            "state": doc["_id"]["state"],
            "year": doc["_id"]["year"],
            "hhi": hhi,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Provider market shares
# ---------------------------------------------------------------------------

def get_provider_shares(db) -> pd.DataFrame:
    """National provider market share by year."""
    pipeline = [
        {"$match": {"wireless_subscribers": {"$exists": True}}},
        {"$group": {
            "_id": {"provider": "$provider", "year": "$year"},
            "total": {"$sum": "$wireless_subscribers"},
        }},
        {"$sort": {"_id.year": 1}},
    ]
    docs = list(db.telecom_subscriptions.aggregate(pipeline))
    df = pd.DataFrame([{
        "provider": d["_id"]["provider"],
        "year": d["_id"]["year"],
        "total": d["total"],
    } for d in docs])

    # Pivot: years as index, providers as columns
    pivot = df.pivot_table(index="year", columns="provider",
                           values="total", fill_value=0)

    # Compute shares
    row_totals = pivot.sum(axis=1)
    shares = pivot.div(row_totals, axis=0) * 100

    # Keep top providers, group rest as "Others"
    avg_share = shares.mean()
    top_providers = avg_share.nlargest(6).index.tolist()
    others = shares.drop(columns=top_providers).sum(axis=1)
    result = shares[top_providers].copy()
    result["Others"] = others

    return result


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_national_wireless(ts: pd.DataFrame, break_dates: list) -> None:
    """Line chart of national wireless subscribers with structural breaks."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts["date"], ts["total_wireless"] / 1e6, linewidth=2, color="#1f77b4")

    labels = ["Mobile Explosion", "Jio Entry"]
    colors = ["orange", "red"]
    for dt, label, color in zip(break_dates, labels, colors):
        ax.axvline(dt, color=color, linestyle="--", linewidth=1.5, alpha=0.8)
        ax.annotate(f"{label}\n({dt.strftime('%b %Y')})",
                    xy=(dt, ax.get_ylim()[1] * 0.85 if color == "red" else ax.get_ylim()[1] * 0.55),
                    fontsize=9, color=color, ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=color, alpha=0.8))

    ax.set_xlabel("Date")
    ax.set_ylabel("Total Wireless Subscribers (millions)")
    ax.set_title("National Wireless Subscriber Growth (2008–2021)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}M"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "obj1_national_wireless.png", dpi=150)
    plt.close(fig)
    print(f"  Saved obj1_national_wireless.png")


def plot_state_growth_ranking(df: pd.DataFrame) -> None:
    """Horizontal bar chart of pre/post CAGR by state."""
    top = df.head(15)
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top))
    bar_h = 0.35
    ax.barh(y_pos + bar_h / 2, top["pre_cagr"] * 100, bar_h, label="Pre-Jio CAGR", color="#4c72b0")
    ax.barh(y_pos - bar_h / 2, top["post_cagr"] * 100, bar_h, label="Post-Jio CAGR", color="#dd8452")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["state"], fontsize=9)
    ax.set_xlabel("CAGR (%)")
    ax.set_title("State-Level Wireless Growth: Pre vs Post Jio Entry")
    ax.legend()
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "obj1_state_growth_ranking.png", dpi=150)
    plt.close(fig)
    print(f"  Saved obj1_state_growth_ranking.png")


def plot_hhi_over_time(hhi_df: pd.DataFrame, break_year: int) -> None:
    """National average HHI over years with competition thresholds."""
    national_hhi = hhi_df.groupby("year")["hhi"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(national_hhi["year"], national_hhi["hhi"], marker="o", linewidth=2, color="#2ca02c")
    ax.axhline(2500, color="red", linestyle=":", alpha=0.6, label="Highly concentrated (>2500)")
    ax.axhline(1500, color="orange", linestyle=":", alpha=0.6, label="Moderately concentrated (>1500)")
    ax.axvline(break_year, color="gray", linestyle="--", alpha=0.5)
    ax.annotate("Jio Entry", xy=(break_year, ax.get_ylim()[1] * 0.9),
                fontsize=9, color="gray", ha="center")
    ax.set_xlabel("Year")
    ax.set_ylabel("HHI (0–10,000 scale)")
    ax.set_title("Market Concentration (HHI) Over Time — National Average")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "obj1_hhi_over_time.png", dpi=150)
    plt.close(fig)
    print(f"  Saved obj1_hhi_over_time.png")


def plot_provider_market_share(shares: pd.DataFrame) -> None:
    """Stacked area chart of provider market shares."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("tab10", n_colors=len(shares.columns))
    ax.stackplot(shares.index, *[shares[col] for col in shares.columns],
                 labels=shares.columns, colors=colors, alpha=0.85)
    ax.set_xlabel("Year")
    ax.set_ylabel("Market Share (%)")
    ax.set_title("Provider Market Share — National Wireless Subscribers")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.set_xlim(shares.index.min(), shares.index.max())
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "obj1_provider_market_share.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved obj1_provider_market_share.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    # 1. National wireless time series
    print("1. Aggregating national wireless subscriber time series...")
    ts = get_national_wireless_ts(db)
    print(f"   {len(ts)} monthly observations ({ts['date'].min().date()} to {ts['date'].max().date()})")

    # 2. Structural break detection (2 breaks: mobile explosion + Jio entry)
    print("\n2. Detecting structural breaks (Bai-Perron / Binseg, n=2)...")
    series = ts["total_wireless"].values
    break_indices = detect_structural_breaks(series, n_bkps=2)
    break_dates = [ts["date"].iloc[min(idx, len(ts) - 1)] for idx in break_indices]
    for i, (idx, dt) in enumerate(zip(break_indices, break_dates)):
        print(f"   Break {i+1}: index {idx} → {dt.strftime('%B %Y')}")

    # Use the second break as the Jio-era break for downstream analysis
    # (first break is the initial mobile explosion ~2011)
    jio_break_idx = break_indices[-1]
    break_date = break_dates[-1]
    break_year = break_date.year
    break_month = break_date.month
    print(f"   Using Break 2 ({break_date.strftime('%B %Y')}) as the Jio-era break")

    # 3. Chow test at both breakpoints
    print("\n3. Chow test at detected breakpoints...")
    for i, (idx, dt) in enumerate(zip(break_indices, break_dates)):
        f_stat, p_val = chow_test(series, idx)
        sig = "***" if p_val < 0.01 else "n.s."
        print(f"   Break {i+1} ({dt.strftime('%b %Y')}): F={f_stat:.2f}, p={p_val:.2e} {sig}")

    # 4. CAGR across three periods
    print("\n4. Computing CAGR across periods...")
    first_break = break_dates[0]
    period1 = ts[ts["date"] < first_break]
    period2 = ts[(ts["date"] >= first_break) & (ts["date"] < break_date)]
    period3 = ts[ts["date"] >= break_date]

    for label, p in [("Period 1 (pre-2011 explosion)", period1),
                     ("Period 2 (2011–Jio entry)", period2),
                     ("Period 3 (post-Jio)", period3)]:
        if len(p) < 2:
            continue
        years = (p["date"].iloc[-1] - p["date"].iloc[0]).days / 365.25
        cagr = compute_cagr(p["total_wireless"].iloc[0], p["total_wireless"].iloc[-1], years)
        print(f"   {label}: {cagr * 100:.2f}%")

    # 5. State-level growth
    print("\n5. Computing state-level growth rates...")
    state_growth = get_state_growth_rates(db, break_year, break_month)
    print(f"   Top 5 states by CAGR acceleration:")
    for _, row in state_growth.head(5).iterrows():
        print(f"     {row['state']:30s}  pre={row['pre_cagr']*100:+.1f}%  post={row['post_cagr']*100:+.1f}%")

    # 6. HHI
    print("\n6. Computing HHI market concentration...")
    hhi_df = compute_hhi(db)
    national_hhi = hhi_df.groupby("year")["hhi"].mean()
    print(f"   HHI range: {national_hhi.min():.0f} – {national_hhi.max():.0f}")
    if break_year in national_hhi.index:
        pre_hhi = national_hhi[national_hhi.index < break_year].mean()
        post_hhi = national_hhi[national_hhi.index >= break_year].mean()
        print(f"   Pre-Jio avg HHI:  {pre_hhi:.0f}")
        print(f"   Post-Jio avg HHI: {post_hhi:.0f}")

    # 7. Provider shares
    print("\n7. Computing provider market shares...")
    shares = get_provider_shares(db)
    print(f"   Providers tracked: {list(shares.columns)}")

    # 8. Generate plots
    print("\n8. Generating visualizations...")
    plot_national_wireless(ts, break_dates)
    plot_state_growth_ranking(state_growth)
    plot_hhi_over_time(hhi_df, break_year)
    plot_provider_market_share(shares)

    client.close()
    print("\nObjective 1 complete.")


if __name__ == "__main__":
    main()

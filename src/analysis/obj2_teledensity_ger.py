import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from linearmodels.panel import PanelOLS

ROOT    = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "db" / "sqlite" / "dsm.db"
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# load + join tele-density and GER into panel
def load_panel():
    con = sqlite3.connect(DB_PATH)

    td = pd.read_sql_query("""
        SELECT s.state_name AS state, t.year, AVG(t.tele_density) AS tele_density
        FROM tele_density t
        JOIN states s USING (state_id)
        GROUP BY s.state_name, t.year
    """, con)

    ger_raw = pd.read_sql_query("""
        SELECT s.state_name AS state, e.year, e.gender, e.category, e.ger
        FROM education_ger e
        JOIN states s USING (state_id)
        WHERE e.ger IS NOT NULL
    """, con)
    con.close()

    total = (ger_raw[(ger_raw.gender == "Total") & (ger_raw.category == "All Categories")]
             [["state", "year", "ger"]].rename(columns={"ger": "ger_total"}))

    female = (ger_raw[(ger_raw.gender == "Female") & (ger_raw.category == "All Categories")]
              [["state", "year", "ger"]].rename(columns={"ger": "ger_female"}))

    scst = (ger_raw[(ger_raw.gender == "Total") &
                    (ger_raw.category.isin(["Scheduled Caste", "Scheduled Tribe"]))]
            .groupby(["state", "year"], as_index=False)["ger"].mean()
            .rename(columns={"ger": "ger_scst"}))

    panel = (td
             .merge(total,  on=["state", "year"], how="inner")
             .merge(female, on=["state", "year"], how="inner")
             .merge(scst,   on=["state", "year"], how="left"))

    return panel.sort_values(["state", "year"]).reset_index(drop=True)


# one-year lag per state
def add_lag(panel):
    panel = panel.copy()
    panel["tele_density_lag1"] = panel.groupby("state")["tele_density"].shift(1)
    return panel.dropna(subset=["tele_density_lag1"]).reset_index(drop=True)


# pearson r per year
def yearly_correlation(panel):
    rows = []
    for year, grp in panel.groupby("year"):
        valid = grp.dropna(subset=["tele_density", "ger_total"])
        if len(valid) >= 5:
            r = valid["tele_density"].corr(valid["ger_total"])
            rows.append({"year": year, "pearson_r": round(r, 3)})
    return pd.DataFrame(rows)


# two-way FE regression
def run_regression(panel, dep_var):
    df = panel.dropna(subset=[dep_var, "tele_density_lag1"]).copy()
    df = df.set_index(["state", "year"])

    res = PanelOLS(
        dependent=df[dep_var],
        exog=df[["tele_density_lag1"]],
        entity_effects=True,
        time_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True)

    return {
        "dep_var":   dep_var,
        "coef":      round(float(res.params["tele_density_lag1"]), 4),
        "std_err":   round(float(res.std_errors["tele_density_lag1"]), 4),
        "t_stat":    round(float(res.tstats["tele_density_lag1"]), 3),
        "p_value":   round(float(res.pvalues["tele_density_lag1"]), 4),
        "r2_within": round(float(res.rsquared_within), 4),
        "n_obs":     int(res.nobs),
        "n_states":  df.index.get_level_values("state").nunique(),
    }


def plot_scatter(panel):
    valid = panel.dropna(subset=["tele_density", "ger_total"])
    m, b  = np.polyfit(valid["tele_density"], valid["ger_total"], 1)
    x     = np.linspace(valid["tele_density"].min(), valid["tele_density"].max(), 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(valid["tele_density"], valid["ger_total"],
               alpha=0.5, edgecolors="k", linewidths=0.4)
    ax.plot(x, m * x + b, color="tomato", linewidth=1.5, label=f"OLS slope = {m:.2f}")
    ax.set_xlabel("Tele-density (phones per 100 people)")
    ax.set_ylabel("GER (Total, All Categories)")
    ax.set_title("Tele-density vs GER (all states, 2013–2021)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "obj2_scatter.png", dpi=150)
    plt.close()
    print("  Saved obj2_scatter.png")


def plot_correlation_over_time(corr_df):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(corr_df["year"], corr_df["pearson_r"],
            marker="o", linewidth=2, color="steelblue")
    ax.axvline(2016, color="tomato", linestyle="--", linewidth=1.2, label="Jio launch (2016)")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Year")
    ax.set_ylabel("Pearson r")
    ax.set_title("Annual Cross-State Correlation: Tele-density vs Total GER")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "obj2_correlation_over_time.png", dpi=150)
    plt.close()
    print("  Saved obj2_correlation_over_time.png")


def plot_coefficients(results):
    labels = {"ger_total": "Total GER", "ger_female": "Female GER", "ger_scst": "SC/ST GER"}
    names  = [labels[r["dep_var"]] for r in results]
    coefs  = [r["coef"] for r in results]
    cis    = [1.96 * r["std_err"] for r in results]
    colors = ["steelblue" if r["p_value"] < 0.05 else "lightgrey" for r in results]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(names, coefs, xerr=cis, color=colors,
            error_kw=dict(ecolor="black", capsize=5), height=0.45)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("β on tele_density (t-1)")
    ax.set_title("Panel Regression Coefficients\n(blue = p < 0.05, grey = not significant)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "obj2_regression_coefficients.png", dpi=150)
    plt.close()
    print("  Saved obj2_regression_coefficients.png")


def get_panel():
    return load_panel()


def get_regression_results():
    panel = add_lag(load_panel())
    return [run_regression(panel, dep) for dep in ["ger_total", "ger_female", "ger_scst"]]


def get_yearly_correlation():
    return yearly_correlation(load_panel())


# check if connectivity narrows gender/caste gaps
def run_equity_gap_regressions(panel):
    con = sqlite3.connect(DB_PATH)
    male = pd.read_sql_query("""
        SELECT s.state_name AS state, e.year, e.ger AS ger_male
        FROM education_ger e
        JOIN states s USING (state_id)
        WHERE e.gender = 'Male' AND e.category = 'All Categories' AND e.ger IS NOT NULL
    """, con)
    con.close()

    panel_full = panel.merge(male, on=["state", "year"], how="left")
    panel_full["gender_gap"] = panel_full["ger_female"] - panel_full["ger_male"]
    panel_full["caste_gap"] = panel_full["ger_total"] - panel_full["ger_scst"]

    results = []
    for dep_var, label in [("gender_gap", "Gender Gap (F-M)"), ("caste_gap", "Caste Gap (Total-SC/ST)")]:
        df = panel_full.dropna(subset=[dep_var, "tele_density_lag1"]).copy()
        if len(df) < 20:
            print(f"  {label}: insufficient data ({len(df)} obs)")
            continue
        df_idx = df.set_index(["state", "year"])
        res = PanelOLS(
            dependent=df_idx[dep_var],
            exog=df_idx[["tele_density_lag1"]],
            entity_effects=True,
            time_effects=True,
        ).fit(cov_type="clustered", cluster_entity=True)

        r = {
            "dep_var": dep_var,
            "label": label,
            "coef": float(res.params["tele_density_lag1"]),
            "std_err": float(res.std_errors["tele_density_lag1"]),
            "t_stat": float(res.tstats["tele_density_lag1"]),
            "p_value": float(res.pvalues["tele_density_lag1"]),
            "r2_within": float(res.rsquared_within),
            "n_obs": int(res.nobs),
        }
        results.append(r)
    return results


# quadratic term to check for diminishing returns
def run_quadratic_regression(panel):
    df = panel.dropna(subset=["ger_total", "tele_density_lag1"]).copy()
    df["td_lag1_sq"] = df["tele_density_lag1"] ** 2
    df_idx = df.set_index(["state", "year"])

    res = PanelOLS(
        dependent=df_idx["ger_total"],
        exog=df_idx[["tele_density_lag1", "td_lag1_sq"]],
        entity_effects=True,
        time_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True)

    return {
        "linear_coef": float(res.params["tele_density_lag1"]),
        "quad_coef": float(res.params["td_lag1_sq"]),
        "linear_p": float(res.pvalues["tele_density_lag1"]),
        "quad_p": float(res.pvalues["td_lag1_sq"]),
        "linear_se": float(res.std_errors["tele_density_lag1"]),
        "quad_se": float(res.std_errors["td_lag1_sq"]),
        "r2_within": float(res.rsquared_within),
        "n_obs": int(res.nobs),
    }


if __name__ == "__main__":
    print("Loading panel...")
    panel     = load_panel()
    panel_lag = add_lag(panel)

    print(f"Panel: {panel['state'].nunique()} states, "
          f"years {panel['year'].min()}–{panel['year'].max()}, "
          f"{len(panel)} obs")
    print(f"Lagged panel: {len(panel_lag)} obs (first year per state dropped)\n")

    corr = yearly_correlation(panel)
    print("Yearly Pearson r (tele-density vs Total GER):")
    print(corr.to_string(index=False))
    print()

    results = []
    for dep in ["ger_total", "ger_female", "ger_scst"]:
        r = run_regression(panel_lag, dep)
        results.append(r)
        sig = ("***" if r["p_value"] < 0.01 else
               "**"  if r["p_value"] < 0.05 else
               "*"   if r["p_value"] < 0.1  else "")
        print(f"{r['dep_var']}:")
        print(f"  β = {r['coef']}  SE = {r['std_err']}  t = {r['t_stat']}  p = {r['p_value']} {sig}")
        print(f"  R²(within) = {r['r2_within']}  N = {r['n_obs']}  States = {r['n_states']}")
        print()

    print("--- NEW: Equity Gap Regressions ---")
    equity_results = run_equity_gap_regressions(panel_lag)
    for r in equity_results:
        sig = "***" if r["p_value"] < 0.01 else "**" if r["p_value"] < 0.05 else "*" if r["p_value"] < 0.1 else ""
        print(f"  {r['label']}:")
        print(f"    beta = {r['coef']:.4f}  SE = {r['std_err']:.4f}  t = {r['t_stat']:.3f}  p = {r['p_value']:.4f} {sig}")
        if r["dep_var"] == "gender_gap":
            # positive coef = female gaining relative to male
            if r["coef"] > 0:
                print(f"    -> Connectivity NARROWS gender gap (women gain relative to men)")
            else:
                print(f"    -> Connectivity WIDENS gender gap")
        else:
            # positive = gap widening (general pulls ahead of SC/ST)
            if r["coef"] > 0:
                print(f"    -> Connectivity WIDENS caste gap (general population benefits more than SC/ST)")
            else:
                print(f"    -> Connectivity NARROWS caste gap (SC/ST catching up)")
    print()

    print("--- NEW: Nonlinearity Test (Quadratic) ---")
    quad = run_quadratic_regression(panel_lag)
    print(f"  Linear coef: {quad['linear_coef']:.4f} (SE={quad['linear_se']:.4f}, p={quad['linear_p']:.4f})")
    print(f"  Quadratic coef: {quad['quad_coef']:.6f} (SE={quad['quad_se']:.6f}, p={quad['quad_p']:.4f})")
    if quad["quad_p"] < 0.05:
        if quad["quad_coef"] < 0:
            print(f"  -> Significant DIMINISHING RETURNS: effect saturates at high connectivity")
            turning_point = -quad["linear_coef"] / (2 * quad["quad_coef"])
            print(f"  -> Turning point at tele-density = {turning_point:.1f}")
        else:
            print(f"  -> Significant ACCELERATING returns")
    else:
        print(f"  -> No significant nonlinearity -- linear model is adequate")

    print("\nGenerating plots...")
    plot_scatter(panel)
    plot_correlation_over_time(corr)
    plot_coefficients(results)
    print(f"\nDone. Outputs saved to {OUT_DIR}")

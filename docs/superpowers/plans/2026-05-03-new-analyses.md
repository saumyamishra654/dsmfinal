# New Analytical Questions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5 new statistical analyses that strengthen the narrative spine: competition→adoption, convergence test, equity gap, leapfrogging, and nonlinearity.

**Architecture:** Single new analysis script `src/analysis/obj6_new_questions.py` that reads from SQLite + the pre-exported `cleaned_datasets/hhi_by_state_year.csv`. Each analysis is a self-contained function that prints results and optionally saves a figure. A second modification adds equity-gap and quadratic regressions to `src/analysis/obj2_teledensity_ger.py`.

**Tech Stack:** Python, pandas, numpy, scipy, statsmodels (PanelOLS), matplotlib, SQLite.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/analysis/obj6_new_questions.py` | Create | Analyses A (HHI→growth), B (convergence), D (leapfrogging) |
| `src/analysis/obj2_teledensity_ger.py` | Modify | Add analyses C (equity gap) and E (nonlinearity/quadratic) |

---

### Task 1: Create `obj6_new_questions.py` — Analysis A: Competition → Adoption

**Files:**
- Create: `src/analysis/obj6_new_questions.py`

- [ ] **Step 1: Write the HHI-to-growth regression**

```python
"""
Objective 6 — New Analytical Questions
A. Did competition (HHI drop) drive tele-density growth?
B. Are lagging states converging (beta-convergence)?
D. Did wireline-leapfrog states benefit more from the Jio shock?
"""
import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "db" / "sqlite" / "dsm.db"
HHI_PATH = ROOT / "cleaned_datasets" / "hhi_by_state_year.csv"
FIGURES = ROOT / "outputs" / "figures"


def get_tele_density_annual():
    """Annual average tele-density per state from SQLite."""
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT s.state_name AS state, t.year, AVG(t.tele_density) AS tele_density
        FROM tele_density t
        JOIN states s USING (state_id)
        GROUP BY s.state_name, t.year
        ORDER BY s.state_name, t.year
    """, con)
    con.close()
    return df


def get_hhi():
    """HHI by state and year from pre-exported CSV."""
    return pd.read_csv(HHI_PATH)


def analysis_a_competition_drives_adoption():
    """
    Regression: did states with bigger HHI drops post-Jio see faster tele-density growth?
    DV: tele-density annual growth rate (2017-2021)
    IV: change in HHI from 2015 to 2018 (the Jio disruption window)
    """
    print("=" * 70)
    print("ANALYSIS A: Did competition (HHI drop) drive tele-density growth?")
    print("=" * 70)

    td = get_tele_density_annual()
    hhi = get_hhi()

    # Compute HHI change: 2015 (pre-Jio) vs 2018 (post-Jio settled)
    hhi_pre = hhi[hhi["year"] == 2015].set_index("state")["hhi"]
    hhi_post = hhi[hhi["year"] == 2018].set_index("state")["hhi"]
    hhi_change = (hhi_post - hhi_pre).rename("hhi_change").reset_index()

    # Compute tele-density growth: CAGR from 2016 to 2021
    td_2016 = td[td["year"] == 2016].set_index("state")["tele_density"]
    td_2021 = td[td["year"] == 2021].set_index("state")["tele_density"]
    td_growth = ((td_2021 / td_2016) ** (1/5) - 1).rename("td_cagr").reset_index()

    # Merge on common states
    merged = hhi_change.merge(td_growth, on="state").dropna()
    print(f"\n  States in regression: {len(merged)}")
    print(f"  HHI change range: {merged['hhi_change'].min():.0f} to {merged['hhi_change'].max():.0f}")
    print(f"  TD CAGR range: {merged['td_cagr'].min()*100:.1f}% to {merged['td_cagr'].max()*100:.1f}%")

    # OLS: td_cagr ~ hhi_change
    slope, intercept, r, p, se = stats.linregress(merged["hhi_change"], merged["td_cagr"])
    print(f"\n  OLS: td_cagr = {intercept:.4f} + {slope:.6f} * hhi_change")
    print(f"  slope = {slope:.6f} (SE={se:.6f})")
    print(f"  R² = {r**2:.4f}")
    print(f"  p-value = {p:.4f} {'***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else 'n.s.'}")
    print(f"\n  Interpretation: A 1000-point HHI drop is associated with {slope*-1000*100:.2f} pp higher CAGR")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(merged["hhi_change"], merged["td_cagr"] * 100, s=60, edgecolors="k", linewidths=0.5)
    x_line = np.linspace(merged["hhi_change"].min(), merged["hhi_change"].max(), 100)
    ax.plot(x_line, (intercept + slope * x_line) * 100, color="tomato", linewidth=2)
    for _, row in merged.iterrows():
        ax.annotate(row["state"], (row["hhi_change"], row["td_cagr"] * 100),
                    fontsize=7, ha="left", va="bottom", xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Change in HHI (2015 → 2018)\n← More competition | Less competition →")
    ax.set_ylabel("Tele-density CAGR 2016–2021 (%)")
    ax.set_title(f"Competition & Adoption: HHI Change vs Tele-density Growth\n(R²={r**2:.3f}, p={p:.3f})")
    ax.axvline(0, color="grey", linestyle=":", alpha=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "obj6_competition_adoption.png", dpi=150)
    plt.close(fig)
    print(f"  Saved obj6_competition_adoption.png")

    return merged
```

- [ ] **Step 2: Run to verify Analysis A works**

Run: `cd /Users/saumyamishra/Desktop/Projects/dsmfinal && python3 -c "from src.analysis.obj6_new_questions import analysis_a_competition_drives_adoption; analysis_a_competition_drives_adoption()"`

---

### Task 2: Analysis B — Beta-Convergence Test

**Files:**
- Modify: `src/analysis/obj6_new_questions.py`

- [ ] **Step 1: Add convergence function**

```python
def analysis_b_convergence():
    """
    Beta-convergence test: do states with lower initial tele-density grow faster?
    Regress: tele-density growth rate (2016-2021) on initial tele-density (2016).
    Negative slope = convergence. Positive = divergence.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS B: Beta-convergence — Is the digital divide closing?")
    print("=" * 70)

    td = get_tele_density_annual()

    td_2016 = td[td["year"] == 2016].set_index("state")["tele_density"].rename("td_initial")
    td_2021 = td[td["year"] == 2021].set_index("state")["tele_density"].rename("td_final")

    merged = pd.concat([td_2016, td_2021], axis=1).dropna()
    merged["cagr"] = (merged["td_final"] / merged["td_initial"]) ** (1/5) - 1
    merged = merged.reset_index()

    print(f"\n  States: {len(merged)}")
    print(f"  Initial TD range: {merged['td_initial'].min():.1f} to {merged['td_initial'].max():.1f}")

    slope, intercept, r, p, se = stats.linregress(merged["td_initial"], merged["cagr"])
    print(f"\n  OLS: cagr = {intercept:.4f} + {slope:.6f} * td_initial")
    print(f"  slope = {slope:.6f} (SE={se:.6f})")
    print(f"  R² = {r**2:.4f}")
    print(f"  p-value = {p:.4f} {'***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else 'n.s.'}")

    if slope < 0 and p < 0.05:
        print(f"\n  RESULT: Significant beta-convergence — lagging states ARE catching up")
    elif slope < 0:
        print(f"\n  RESULT: Weak convergence (negative slope but not significant)")
    elif slope > 0 and p < 0.05:
        print(f"\n  RESULT: Significant DIVERGENCE — the gap is WIDENING")
    else:
        print(f"\n  RESULT: No significant convergence or divergence")

    # Half-life of convergence (if converging)
    if slope < 0:
        half_life = -np.log(2) / slope if slope != 0 else float("inf")
        print(f"  Half-life of convergence: {half_life:.1f} tele-density units")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(merged["td_initial"], merged["cagr"] * 100, s=60, edgecolors="k", linewidths=0.5)
    x_line = np.linspace(merged["td_initial"].min(), merged["td_initial"].max(), 100)
    ax.plot(x_line, (intercept + slope * x_line) * 100, color="tomato", linewidth=2)
    for _, row in merged.iterrows():
        ax.annotate(row["state"], (row["td_initial"], row["cagr"] * 100),
                    fontsize=7, ha="left", va="bottom", xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Initial Tele-density (2016)")
    ax.set_ylabel("CAGR in Tele-density, 2016–2021 (%)")
    ax.set_title(f"Beta-Convergence Test: Initial Level vs Growth\n(slope={slope:.5f}, p={p:.3f})")
    ax.axhline(0, color="grey", linestyle=":", alpha=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "obj6_convergence.png", dpi=150)
    plt.close(fig)
    print(f"  Saved obj6_convergence.png")

    return merged
```

- [ ] **Step 2: Run to verify**

---

### Task 3: Analysis D — Wireline Leapfrogging

**Files:**
- Modify: `src/analysis/obj6_new_questions.py`

- [ ] **Step 1: Add leapfrogging function**

```python
def analysis_d_leapfrogging():
    """
    Did states that skipped wireline (direct-to-wireless) grow faster post-Jio?
    Compute wireless/(wireless+wireline) ratio in 2017 as 'leapfrog index'.
    Regress tele-density growth on leapfrog index.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS D: Wireline Leapfrogging — Did skip-wireline states grow faster?")
    print("=" * 70)

    con = sqlite3.connect(DB_PATH)
    ww = pd.read_sql_query("""
        SELECT s.state_name AS state, w.year,
               AVG(w.wireless_millions) AS wireless,
               AVG(w.wireline_millions) AS wireline
        FROM wired_wireless w
        JOIN states s USING (state_id)
        GROUP BY s.state_name, w.year
    """, con)
    con.close()

    # Leapfrog index: wireless share in 2017 (first year of wired_wireless data)
    ww_2017 = ww[ww["year"] == 2017].copy()
    ww_2017["leapfrog_idx"] = ww_2017["wireless"] / (ww_2017["wireless"] + ww_2017["wireline"])
    leapfrog = ww_2017[["state", "leapfrog_idx"]].set_index("state")

    # Growth in wireless subscribers 2017-2023
    ww_first = ww[ww["year"] == 2017].set_index("state")["wireless"]
    ww_last = ww[ww["year"] == ww["year"].max()].set_index("state")["wireless"]
    years = ww["year"].max() - 2017
    growth = ((ww_last / ww_first) ** (1/years) - 1).rename("wireless_cagr")

    merged = leapfrog.join(growth).dropna().reset_index()
    print(f"\n  States: {len(merged)}")
    print(f"  Leapfrog index range: {merged['leapfrog_idx'].min():.3f} to {merged['leapfrog_idx'].max():.3f}")
    print(f"  (1.0 = pure wireless, 0.5 = equal wireline/wireless)")

    slope, intercept, r, p, se = stats.linregress(merged["leapfrog_idx"], merged["wireless_cagr"])
    print(f"\n  OLS: wireless_cagr = {intercept:.4f} + {slope:.4f} * leapfrog_idx")
    print(f"  slope = {slope:.4f} (SE={se:.4f})")
    print(f"  R² = {r**2:.4f}")
    print(f"  p-value = {p:.4f} {'***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else 'n.s.'}")

    if slope > 0 and p < 0.1:
        print(f"\n  RESULT: Pure-wireless states grew FASTER — leapfrogging advantage confirmed")
    elif slope < 0 and p < 0.1:
        print(f"\n  RESULT: States with wireline infrastructure grew faster — no leapfrog advantage")
    else:
        print(f"\n  RESULT: No significant relationship between wireline presence and growth")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(merged["leapfrog_idx"] * 100, merged["wireless_cagr"] * 100, s=60, edgecolors="k", linewidths=0.5)
    x_line = np.linspace(merged["leapfrog_idx"].min(), merged["leapfrog_idx"].max(), 100)
    ax.plot(x_line * 100, (intercept + slope * x_line) * 100, color="tomato", linewidth=2)
    for _, row in merged.iterrows():
        ax.annotate(row["state"], (row["leapfrog_idx"] * 100, row["wireless_cagr"] * 100),
                    fontsize=7, ha="left", va="bottom", xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Wireless Share of Total Subscribers in 2017 (%)\n(higher = more 'leapfrogged')")
    ax.set_ylabel("Wireless Subscriber CAGR, 2017–2023 (%)")
    ax.set_title(f"Wireline Leapfrogging: Did Pure-Wireless States Grow Faster?\n(R²={r**2:.3f}, p={p:.3f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "obj6_leapfrogging.png", dpi=150)
    plt.close(fig)
    print(f"  Saved obj6_leapfrogging.png")

    return merged
```

- [ ] **Step 2: Run to verify**

---

### Task 4: Add main() and run all three

**Files:**
- Modify: `src/analysis/obj6_new_questions.py`

- [ ] **Step 1: Add main block**

```python
def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    analysis_a_competition_drives_adoption()
    analysis_b_convergence()
    analysis_d_leapfrogging()
    print("\n" + "=" * 70)
    print("Objective 6 complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run full script**

Run: `cd /Users/saumyamishra/Desktop/Projects/dsmfinal && python3 src/analysis/obj6_new_questions.py`

---

### Task 5: Analysis C — Equity Gap (modify obj2)

**Files:**
- Modify: `src/analysis/obj2_teledensity_ger.py`

- [ ] **Step 1: Add equity gap regression function**

Add after the existing `run_regression` function:

```python
def run_equity_gap_regressions(panel):
    """
    Does connectivity NARROW the gender/caste gap?
    DV: (female_GER - male_GER) or (total_GER - scst_GER)
    Positive coefficient = connectivity closes the gap.
    """
    # Load male GER to compute gender gap
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
```

- [ ] **Step 2: Add quadratic/nonlinearity regression (Analysis E)**

```python
def run_quadratic_regression(panel):
    """
    Does tele-density have diminishing returns on GER?
    Add tele_density_lag1^2 to the panel regression.
    Negative quadratic coefficient = saturation/diminishing returns.
    """
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
        "r2_within": float(res.rsquared_within),
        "n_obs": int(res.nobs),
    }
```

- [ ] **Step 3: Add to main block and run**

Add a section at the end of obj2's `if __name__ == "__main__":` block:

```python
    print("\n--- NEW: Equity Gap Regressions ---")
    equity_results = run_equity_gap_regressions(panel_lag)
    for r in equity_results:
        sig = "***" if r["p_value"] < 0.01 else "**" if r["p_value"] < 0.05 else "*" if r["p_value"] < 0.1 else ""
        print(f"  {r['label']}:")
        print(f"    beta = {r['coef']:.4f}  SE = {r['std_err']:.4f}  t = {r['t_stat']:.3f}  p = {r['p_value']:.4f} {sig}")
        if r['coef'] > 0:
            print(f"    -> Connectivity NARROWS the gap")
        else:
            print(f"    -> Connectivity WIDENS the gap")

    print("\n--- NEW: Nonlinearity Test (Quadratic) ---")
    quad = run_quadratic_regression(panel_lag)
    print(f"  Linear coef: {quad['linear_coef']:.4f} (p={quad['linear_p']:.4f})")
    print(f"  Quadratic coef: {quad['quad_coef']:.6f} (p={quad['quad_p']:.4f})")
    if quad['quad_p'] < 0.05:
        if quad['quad_coef'] < 0:
            print(f"  -> Significant DIMINISHING RETURNS: effect saturates at high connectivity")
            turning_point = -quad['linear_coef'] / (2 * quad['quad_coef'])
            print(f"  -> Turning point at tele-density = {turning_point:.1f}")
        else:
            print(f"  -> Significant ACCELERATING returns")
    else:
        print(f"  -> No significant nonlinearity — linear model is adequate")
```

- [ ] **Step 4: Run modified obj2**

Run: `cd /Users/saumyamishra/Desktop/Projects/dsmfinal && python3 src/analysis/obj2_teledensity_ger.py`

---

### Task 6: Commit all new analysis code

- [ ] **Step 1: Stage and commit**

```bash
git add src/analysis/obj6_new_questions.py src/analysis/obj2_teledensity_ger.py
git commit -m "feat: add convergence, competition, equity gap, leapfrog, and nonlinearity analyses"
```

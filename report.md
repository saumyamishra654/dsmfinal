# Digital India: Telecom, Education & Payments

**CS-3510: Data Science and Management — Final Project**
by Saumya Mishra & Vatsl Goswami

---

## 1. Introduction

Over the past decade, India has undergone one of the fastest digital transformations in the world. Today, over 1.17 billion people hold a wireless telecom subscription. UPI payments have overtaken debit cards and cash as the dominant mode of transaction, processing over 5.86 billion transactions monthly. The simultaneous launch of Jio's free 4G service (September 2016) and UPI's interoperable payment rail (August 2016) created a flywheel effect of extraordinary magnitude.

This project investigates India's digital journey using six public datasets spanning telecom subscriptions, tele-density, education enrolment, digital payments, electricity consumption, and provider-level market data. Our central thesis:

> **India's telecom market underwent a verifiable structural transformation in 2016. The digital divide between states was narrowing during the preceding growth era — but froze after Jio-era consolidation, locking in spatial inequality that current growth rates cannot resolve.**

### 1.1 Research Questions

- **RQ1 (The Shock):** How did India's wireless subscriber base grow between 2008-2023, and can we formally identify structural breaks using methods that account for data-driven breakpoint selection?
- **RQ2 (Connectivity & Education):** Does state-level tele-density correlate with Gross Enrolment Ratio (GER), and does the effect differ for women and SC/ST groups?
- **RQ3 (Digital Payments):** Did wireless growth precede digital transaction growth (Granger sense)? How did payment method composition shift?
- **RQ4 (The Digital Divide):** Which states are digitally lagging? How do they cluster, and how long would convergence take?
- **RQ5 (Convergence):** Were lagging states catching up pre-Jio? Did convergence continue or halt after the shock?
- **RQ6 (Equity):** Has telecom expansion improved SC/ST educational inclusion relative to the general population?

> **[POST-FEEDBACK CHANGE]:** RQ5 and RQ6 were added after initial presentation feedback that the original questions were "scattered and directionless." The narrative was restructured from four independent questions into a causal chain: shock -> mechanism -> outcome -> inequality. This gives the project a unified argument rather than four disconnected mini-analyses.

---

## 2. Data & Methodology

### 2.1 Data Sources

| Dataset | Source | Rows | Period | Granularity |
|---------|--------|------|--------|-------------|
| Area-wise Tele-density | TRAI | 1,800 | 2013-2022 | State x Month |
| Wired & Wireless Subscribers | TRAI | 1,386 | 2017-2023 | State x Month |
| Gross Enrolment Ratio | AISHE / MoE | 3,231 | 2012-2021 | State x Year x Gender x Category |
| Digital Transactions | RBI | 64 | 2016-2021 | National x Month |
| Sector-wise Electricity | CEA | 158 | 1970-2023 | National x Year x Sector |
| Provider-level Subscriptions | TRAI (via MongoDB) | 58,533 | 2008-2021 | State x Provider x Month |

### 2.2 Pipeline

Raw CSVs are cleaned through `data_cleaning.py` and saved as type-preserving intermediate files. Both database loaders read from these, so the SQL and Mongo databases always agree.

- **SQLite** stores the relational, panel-shaped data in 3NF normalization. `state_name` is stored exactly once in a `states` table; all other tables reference it via the integer `state_id` FK. This eliminates redundancy and makes state-name updates trivial.
- **MongoDB** stores provider-level monthly subscription data (~58K documents). A document model is more natural here because each month has 6-10 providers per circle with varying field presence. This was later flattened into SQLite for the deployed web version.

### 2.3 Statistical Methods

| Method | Purpose | Key Assumption |
|--------|---------|----------------|
| Bai-Perron (Binseg, L2) | Detect breakpoints in national wireless series | Piecewise linear structure |
| Quandt-Andrews sup-Wald | Inference on breakpoints without pre-specification | Linear model under null |
| Two-way FE Panel Regression | Within-state effect of tele-density on GER | Strict exogeneity |
| Clustered SEs + Wild Bootstrap | Valid inference with 18 clusters | Cluster-level dependence |
| Beta-convergence (log-spec) | Test if lagging states grow faster | Log-linear relationship |
| Spearman / Kendall | Non-parametric robustness for convergence | Monotonic relationship |
| HC3 robust SEs | Address heteroskedasticity | Leverage-weighted residuals |
| Granger causality (VAR F-test) | Predictive precedence of wireless on payments | Stationarity |
| STL Decomposition | Separate trend/seasonal/residual in payments | Additive decomposition |
| Louvain community detection | Identify natural state clusters | Cosine-similarity graph |
| K-means + silhouette | Cluster validation | Euclidean distance |
| Gap analysis | Project years to convergence | Constant growth rates |
| Wild-cluster bootstrap | Small-sample panel inference | Rademacher weights |

> **[POST-FEEDBACK CHANGE]:** The Quandt-Andrews sup-Wald test, log-specification convergence, HC3/Spearman/Kendall/LOO robustness checks, wild-cluster bootstrap, and the ratio-based equity measure were all added after external statistical audit identified: (a) circular inference in Chow tests, (b) small-cluster bias with 18 states, (c) construct-validity issues in the original gap measure.

---

## 3. Challenges & Design Decisions

### 3.1 The Telecom Circle vs State Mismatch

The biggest challenge was realizing that telecom data uses TRAI geographical 'circles', not state boundaries:

- Kolkata and Mumbai are separate circles (not part of West Bengal or Maharashtra). Merging would double-count.
- 'North East' is a single circle covering 7 states.
- Chhattisgarh, Jharkhand, Uttarakhand, Telangana, and Goa are folded into older parent-state circles.

We use only the 17 major states where TRAI circle names correspond directly with state names. This sacrifices coverage for accuracy.

### 3.2 Financial Year vs Calendar Year

Telecom data uses Indian Financial Year (Apr-Mar). Digital transaction data uses Calendar Year. We parse both formats and align at the month level. FY 2023 month 'April' = CY 2022-04, because Indian FY 2023 spans April 2022 to March 2023.

### 3.3 Why Surrogate Keys for States

Instead of storing state names as strings in every row, we use a foreign integer key with `state_id`. This is standard 3NF normalization -- eliminates redundancy, enables single-point state name updates, and reduces storage.

### 3.4 Post-Jio Tele-density Decline

> **[POST-FEEDBACK CHANGE]:** During analysis, we discovered that tele-density *declines* in 15/18 states from 2016-2021 (CAGR range: -4.1% to +1.0%). This is NOT because fewer people have phones -- it's because tele-density counts SIM cards per 100 people, and after Jio made single-plan subscriptions dominant, people dropped secondary/tertiary SIMs. Raw wireless subscriber counts continue growing in all states (3.9%-10.1% CAGR 2017-2023). We therefore use tele-density for the pre-Jio period (where it validly measures growth) and raw subscribers for post-Jio analysis.

---

## 4. RQ1 -- The Exogenous Shock

### 4.1 Background

On September 5, 2016, Reliance Jio launched with free 4G data for all subscribers. Within six months, Jio acquired over 100 million subscribers -- the fastest-growing mobile operator in history. Incumbents (Airtel, Vodafone, Idea) were forced into a price war that saw data costs fall by over 95% within two years. Average monthly data consumption jumped from under 1 GB to over 11 GB per user.

But was this a *structural break* in the statistical sense? Or was growth already accelerating, with Jio merely riding an existing wave?

### 4.2 Structural Break Detection (Quandt-Andrews Sup-Wald)

We applied the Quandt-Andrews sup-Wald test to the national monthly wireless subscriber time series (149 observations, 2008-2021). This tests ALL possible breakpoints in the interior 70% of the sample and takes the maximum F-statistic, comparing to Andrews (1993) critical values that correct for the data-driven search.

| Test | Result | Critical Value (1%) | Verdict |
|------|--------|---------------------|---------|
| Break 1 (September 2011) | sup-F = 253.60 | 12.35 | Significant |
| Break 2 (November 2016) | sup-F = 131.59 | 12.35 | Significant |
| Bootstrap p-value (999 iter) | < 0.001 | -- | Confirmed |

Both breaks exceed the 1% critical value by more than an order of magnitude. The sup-F of 253.60 is among the most decisive structural break results in applied telecommunications economics.

> **[POST-FEEDBACK CHANGE]:** The original analysis used only Chow tests at the Bai-Perron-detected breakpoints. An external audit flagged this as circular inference (breakpoint found from data, then tested as if known). The Quandt-Andrews sup-Wald test was added to address this -- it accounts for the data-driven search and provides valid p-values without requiring pre-specification of the break date.

### 4.3 Three Growth Phases

The structural breaks divide the series into three distinct phases:

1. **Rapid Expansion (2008-2011):** Driven by falling handset costs and regulatory liberalization. CAGR ~35%.
2. **Plateau (2011-2016):** Growth slowed as the "easy" urban market saturated and the 2G spectrum controversy depressed investment. CAGR ~5%.
3. **Jio Acceleration (2016-2021):** Cheap data drove rural and secondary-device adoption. Monthly additions tripled from ~4.2M to ~12.8M subscribers.

### 4.4 Market Concentration (HHI)

The Herfindahl-Hirschman Index measures market concentration on a 0-10,000 scale (>2,500 = highly concentrated, <1,500 = competitive).

- **Pre-Jio HHI:** ~2,500 (moderately concentrated oligopoly: Airtel, Vodafone, Idea dominated)
- **Post-Jio HHI:** ~1,800 (competitive market)
- **Paradox:** HHI decreased even as the number of major operators fell from ~12 to 4. Jio captured share from the combined tail of smaller operators, creating a more evenly distributed market.

### 4.5 Provider Market Share Evolution

Jio rose from 0% to ~35% market share within three years. Vodafone and Idea merged (forming Vi) but continued losing share. BSNL became increasingly marginal. By 2021, the market was effectively a triopoly (Jio ~35%, Airtel ~30%, Vi ~25%).

---

## 5. RQ5 -- Convergence & Stasis

> **This is the strongest new finding in the project.**

### 5.1 The Convergence Hypothesis

Beta-convergence is a foundational concept in growth economics: do entities that start behind grow faster, thereby "catching up" over time? In telecommunications, we ask: were states with lower initial tele-density growing their networks faster?

This is not merely academic. If convergence holds, the digital divide is self-correcting through market forces. If convergence fails, active policy intervention becomes necessary.

### 5.2 Pre-Jio Convergence (2013-2016)

We regress CAGR on log(initial tele-density) -- the standard Barro & Sala-i-Martin specification. A negative slope indicates convergence.

**Results:**

| Method | Statistic | p-value |
|--------|-----------|---------|
| OLS (log-spec) | slope = -0.039 | **0.030** |
| HC3 robust SEs | -- | **0.003** |
| Spearman rank | rho = -0.565 | **0.015** |
| Kendall tau | tau = -0.412 | **0.017** |
| Leave-one-out | All 18 slopes negative | All p < 0.10 |

All growth rates are positive (4.3% to 14.0% CAGR). States like Jammu & Kashmir (initial TD: 66.8, CAGR: 14.0%) and Himachal Pradesh (initial TD: 105.6, CAGR: 12.9%) grew rapidly from low bases, while Delhi (initial TD: 226.8, CAGR: 4.5%) grew slowly from a high base. This is textbook beta-convergence.

The result is robust across parametric (OLS, HC3), non-parametric (Spearman, Kendall), and sensitivity (LOO) methods. No single state drives the finding -- dropping any one state keeps all slopes negative.

### 5.3 Post-Jio: No Convergence (2017-2022)

| Method | Statistic | p-value |
|--------|-----------|---------|
| OLS (log-spec) | slope = +0.005 | 0.667 |
| Spearman rank | rho = -0.187 | 0.458 |

The convergence coefficient collapses from -0.039 to +0.005 and loses all significance. The confidence interval straddles zero. States that were behind in 2017 did not grow faster. The catch-up process halted.

### 5.4 Why Did Convergence Stop?

We propose three complementary mechanisms:

1. **Universal Shock, Uniform Impact:** Jio offered the same free/cheap data everywhere simultaneously. Unlike the pre-Jio era where lagging states had more "room to grow," Jio's blanket disruption lifted all states roughly equally, preserving existing gaps.
2. **Infrastructure Constraints Bind:** After "easy" urban subscribers were acquired, further growth requires physical tower infrastructure in rural areas. Lagging states often lack this infrastructure.
3. **SIM Consolidation:** Multi-SIM users (predominantly in high-TD states) dropped extra SIMs, artificially depressing measured tele-density growth in leaders while not affecting laggards. This masks any remaining convergence in the per-capita metric.

### 5.5 Policy Implication

If the digital divide is no longer self-correcting through market forces, targeted interventions (BharatNet rural fiber, mandatory rollout obligations, device subsidies) become essential for preventing permanent stratification.

> **[POST-FEEDBACK CHANGE]:** This entire section was added after feedback. The original analysis had a "gap analysis" showing years-to-close but never formally tested whether convergence was occurring. The log-specification, multiple robustness methods, and period comparison are all new contributions.

---

## 6. RQ2 -- Connectivity & Education

### 6.1 The Cross-State Story: A Strong Correlation

The Pearson correlation between tele-density and Total GER is r ~ 0.77 across all state-years (2013-2021). This is one of the strongest bivariate relationships in our dataset. States with higher connectivity consistently show higher educational participation.

The relationship is intuitive: mobile internet enables access to educational resources, online courses, distance learning, and information about opportunities. The correlation strengthens slightly over time (from r=0.80 in 2013 to r=0.71 in 2021), remaining consistently high throughout.

### 6.2 Panel Regression: The Within-State Challenge

Cross-sectional correlations are suggestive but not causal. Rich, urban, well-governed states naturally have both more cell towers and more universities. The critical test is whether *changes* in tele-density within a state predict *changes* in GER, holding constant all time-invariant state characteristics.

We estimate a two-way fixed effects model (state FE + year FE), clustering standard errors by state. Tele-density is lagged one year to reduce simultaneity.

| Dependent | Beta | SE | t | p | R-sq(within) | N |
|-----------|------|----|----|---|-----------|---|
| Total GER | 0.082 | 0.056 | 1.47 | 0.145 | 0.248 | 143 |
| Female GER | 0.073 | 0.092 | 0.79 | 0.429 | 0.198 | 143 |
| SC/ST GER | -0.019 | 0.076 | -0.25 | 0.801 | -0.057 | 143 |

**Interpretation:** The coefficient for Total GER is positive and economically meaningful -- a 10-point increase in tele-density is associated with a 0.82 pp increase in GER. However, it is NOT statistically significant at conventional levels (p=0.145). With 18 states over ~8 years (143 observations after lagging), we lack statistical power to detect what may be a real but modest within-state effect. The confidence interval [-0.028, +0.192] includes both zero and meaningful positive values.

The disconnect between cross-sectional (r=0.77) and panel (p=0.145) results highlights the importance of confounders: state-level development drives both outcomes simultaneously, and our fixed effects absorb this -- but may also absorb part of the true effect.

### 6.3 Yearly Correlation Trend

| Year | Pearson r |
|------|-----------|
| 2013 | 0.796 |
| 2015 | 0.812 |
| 2017 | 0.777 |
| 2019 | 0.721 |
| 2021 | 0.708 |

The correlation weakens slightly post-Jio, possibly because tele-density becomes less discriminating as connectivity becomes universal -- but remains consistently high (>0.70).

---

## 7. RQ6 -- The Equity Question

### 7.1 Construct: SC/ST Inclusion Ratio

> **[POST-FEEDBACK CHANGE]:** Originally used `ger_total - ger_scst` (a difference) as the dependent variable. An external audit flagged a construct-validity issue: `ger_total` includes SC/ST enrollments in its denominator, so the difference is not cleanly "general vs SC/ST." Reframed as a ratio: `inclusion_ratio = ger_scst / ger_total`. Values closer to 1 indicate more equal outcomes. A negative coefficient on tele-density means connectivity is associated with SC/ST falling further behind.

### 7.2 Results

| Method | Coefficient | p-value | Verdict |
|--------|-------------|---------|---------|
| Panel FE (asymptotic clustered SE) | -0.0032 | 0.057 | Borderline |
| Wild-cluster bootstrap (999 iter, Rademacher) | -- | 0.125 | Not significant |

The asymptotic result is suggestive (p=0.057) -- SC/ST inclusion appears to decrease with connectivity. But the wild-cluster bootstrap, which corrects for the well-documented downward bias of clustered SEs with few clusters (18), produces p=0.125. This does NOT survive small-sample correction.

> **[POST-FEEDBACK CHANGE]:** Wild-cluster bootstrap was added because the audit identified that standard clustered SEs are downward-biased with <20 clusters, potentially producing false positives. With the correction, we cannot claim connectivity harms SC/ST inclusion.

### 7.3 Interpretation

We cannot reject the null that connectivity benefits SC/ST populations equally. The suggestive negative direction warrants further investigation with larger panels, better identification (e.g., instrumental variables from 4G tower rollout timing), or state-level SC/ST-specific connectivity data.

---

## 8. RQ3 -- Digital Payments

### 8.1 The UPI Phenomenon

UPI was launched in August 2016, nearly simultaneously with Jio. Between April 2017 and July 2022, UPI transaction volume grew from 0.7 crore to over 586 crore transactions monthly -- an 83,700% increase in five years. This simultaneously cannibalized debit card transactions, which stagnated while UPI skyrocketed.

### 8.2 Granger Causality

We test whether wireless subscriber growth (month-on-month) has predictive precedence over digital transaction growth using VAR-based Granger causality at lags 1-4 months. Both series are first-differenced to growth rates and confirmed stationary via ADF tests.

**Result:** At the monthly frequency, wireless growth does not significantly Granger-cause digital payment growth at any lag. This indicates the link is structural/long-run rather than month-to-month. Given the limited sample (~60 usable observations after differencing), statistical power is low.

**Caveat:** Granger "causality" is predictive precedence, not structural causation. Even a significant result would not prove that more phones cause more UPI transactions -- only that phone growth helps predict transaction growth.

### 8.3 STL Decomposition

Seasonal-Trend Decomposition (STL, period=12) of the digital transaction series reveals:

- **Trend:** Continuous upward, with COVID-19 as a temporary V-shaped dip (Mar-May 2020) that recovered within months. The trend was already rising pre-COVID; the pandemic accelerated but did not create the digital shift.
- **Seasonal:** Modest (~10% amplitude). Peak months align with festivals and year-end (October-December).
- **Residual:** Well-behaved with no systematic patterns.

### 8.4 Payment Method Composition

UPI/BHIM went from negligible market share to dominant within four years:

- 2017: UPI ~2%, Debit Cards ~22%, Other Digital ~76%
- 2019: UPI ~30%, Debit Cards ~15%, Other ~55%
- 2021: UPI ~60%, Debit Cards ~5%, Other ~35%

COVID-19 created a further step-change that did not revert, suggesting a permanent behavioral shift. Debit card usage effectively collapsed to near-zero share.

---

## 9. RQ4 -- The Digital Divide

### 9.1 Feature Construction

For each state, we build a 4-feature vector:
1. Mean tele-density (2013-2022)
2. Tele-density growth slope (linear trend)
3. Mean GER (2012-2021)
4. GER growth slope

Features are standardized (z-scored) before clustering.

### 9.2 Clustering Methods

**K-means** (silhouette-optimized): Best k=2-3 depending on random seed. Silhouette score ~0.45.

**Louvain community detection:** We build a cosine-similarity graph (all positive edges), then apply the Louvain algorithm. This identifies natural groupings without requiring k to be pre-specified:

- **Leaders:** Delhi, Himachal Pradesh, Karnataka, Kerala, Tamil Nadu, Punjab (high mean TD, high GER)
- **Mid-tier:** Haryana, Jammu & Kashmir, Maharashtra, Andhra Pradesh, Gujarat (moderate on both)
- **Laggards:** Bihar, Assam, Madhya Pradesh, Odisha, Rajasthan, Uttar Pradesh, West Bengal (low TD, low GER)

### 9.3 PCA Biplot

The first two principal components capture ~72% of variance. PC1 loads on mean tele-density and mean GER (general "digital development"). PC2 loads on growth slopes (trajectory). The biplot clearly separates the three communities.

### 9.4 Gap Analysis

For each lagging state, we compute: (leader community mean TD - current state TD) / annual growth rate = years to close gap.

| State | Mean TD | Annual Growth | Gap to Leader | Years to Close |
|-------|---------|---------------|---------------|----------------|
| Bihar | ~57 | 0.39 | ~70 | ~135 |
| Rajasthan | ~76 | 0.15 | ~50 | ~287 |
| Uttar Pradesh | ~70 | 0.95 | ~57 | ~60 |
| Madhya Pradesh | ~78 | 1.20 | ~49 | ~41 |
| West Bengal | ~85 | 1.05 | ~42 | ~40 |

The convergence test (Section 5) formally confirms these projections are realistic -- current growth rates are indeed uniform across states, so these gaps will not close organically.

### 9.5 Spatial Co-location of Exclusion

The same states that lag on connectivity also lag on education enrollment. The correlation between cluster membership and GER is near-perfect. Digital exclusion and educational exclusion are spatially co-located -- reinforcing each other through the mechanisms explored in Section 6.

---

## 10. Conclusion

### 10.1 Summary of Findings

| Finding | Strength | Key Statistic |
|---------|----------|---------------|
| Structural breaks (2011, 2016) | Rock solid | sup-F = 253 (crit = 12.35) |
| Pre-Jio convergence | Robust | p = 0.003 (HC3) |
| Post-Jio convergence froze | Robust null | p = 0.667 |
| Cross-state TD-GER correlation | Strong descriptive | r = 0.77 |
| Within-state TD->GER (panel FE) | Inconclusive | p = 0.145 |
| SC/ST equity effect | Suggestive, not robust | bootstrap p = 0.125 |
| Bihar gap projection | Descriptive | ~135 years |
| HHI collapse | Descriptive | 2500 -> 1800 |

### 10.2 What We Built

- A `data_cleaning.py` module producing type-preserving intermediate files from 6 raw datasets.
- A SQLite schema in 3NF with FK constraints across 7 tables (including migrated telecom_subscriptions).
- A MongoDB store for provider-level monthly subscription data (58,533 documents).
- Six analysis modules (`obj1` through `obj6`) with reproducible figure outputs.
- A Streamlit dashboard with interactive Plotly charts and LLM-powered data chat.
- A Next.js report site deployed on Vercel + FastAPI backend on Render with interactive data explorer.

### 10.3 Limitations

1. **Small panel:** Only 17-18 states due to TRAI circle mapping. This limits statistical power for panel inference.
2. **No state-level digital payments:** Transaction data is national only -- cannot compare adoption across states.
3. **Small-cluster inference:** 18 clusters is borderline for clustered SEs. Wild-cluster bootstrap addresses this but at cost of power.
4. **No time-varying controls:** GDP, urbanization, education spending are not controlled for -- omitted variable bias is possible in panel regressions.
5. **Tele-density is not internet access:** It counts SIM cards per 100 people, not unique users with meaningful connectivity.
6. **Multiple testing:** Only the convergence result survives all robustness checks. The caste equity finding fails Bonferroni correction (threshold 0.0083, our p = 0.0099/0.057).
7. **Granger is not causation:** Predictive precedence does not establish structural causal mechanisms.
8. **GER data ends in 2021:** Post-Jio education analysis is limited to a 5-year window.

### 10.4 What This Means

India's digital transformation has been faster than almost anywhere else in the world. But it has not been even. The convergence that was occurring pre-Jio stopped after market consolidation. The same states that lag on connectivity also lag on education enrolment, and at current growth rates that gap is not going to close in any reasonable time horizon.

Connectivity expansion alone is not enough. The real policy question is not whether to keep expanding connectivity, but how to ensure lagging states benefit at the same rate as leaders -- through targeted infrastructure investment, mandatory rural rollout obligations, or direct device and data subsidies. Without such interventions, the digital divide will remain structurally locked for generations.

---

## References & Data Citations

### Datasets

1. **Telecom Regulatory Authority of India (TRAI).** "Indian Telecom Services Performance Indicators." Quarterly reports, 2008-2023. Available at: https://www.trai.gov.in/release-publication/reports/performance-indicators-reports

2. **Telecom Regulatory Authority of India (TRAI).** "The Indian Telecom Services Performance Indicators (Area-wise Tele-density)." 2013-2023. Available at: https://www.trai.gov.in/release-publication/reports/performance-indicators-reports

3. **Ministry of Education, Government of India.** "All India Survey on Higher Education (AISHE)." Annual reports, 2012-2021. Available at: https://aishe.gov.in/

4. **Reserve Bank of India (RBI).** "Digital Payments Statistics." Monthly data, 2016-2021. Available at: https://www.rbi.org.in/Scripts/Statistics.aspx

5. **Central Electricity Authority (CEA).** "Growth of Electricity Sector in India from 1947-2023." Available at: https://cea.nic.in/

### Statistical Methods

6. Andrews, D. W. K. (1993). "Tests for Parameter Instability and Structural Change with Unknown Change Point." *Econometrica*, 61(4), 821-856.

7. Barro, R. J., & Sala-i-Martin, X. (1992). "Convergence." *Journal of Political Economy*, 100(2), 223-251.

8. Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). "Bootstrap-Based Improvements for Inference with Clustered Errors." *Review of Economics and Statistics*, 90(3), 414-427.

9. Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008). "Fast unfolding of communities in large networks." *Journal of Statistical Mechanics*, P10008.

### Software

10. Python 3.12, pandas, numpy, scipy, statsmodels, scikit-learn, networkx, ruptures, plotly, streamlit, FastAPI, Next.js.

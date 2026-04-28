# Presentation Speech — Digital India: Telecom, Education & Payments

> Navigate through pages in the Streamlit sidebar as you speak. Each section below corresponds to a page.
>
> **SAUMYA's sections:** PAGE 0 (shared intro), PAGE 1, PAGE 3, LLM DEMO, CLOSING
> **VATSL's sections:** PAGE 2, PAGE 4, PAGE 5

---

## PAGE 0: Study Overview (click "Study Overview" in sidebar) — SHARED INTRO

Good morning everyone. I'm Saumya, and along with my partner Vatsl, we've been investigating India's digital transformation — specifically, how the expansion of telecom infrastructure over the last decade has shaped education access, digital payments, and the gap between digitally advanced and lagging states.

Our research question is: **How has India's telecom infrastructure expansion — particularly the wireless mobile revolution and Jio's entry — driven digital financial inclusion and higher education access across states?**

We worked with five national datasets from NDAP and TRAI, covering over 65,000 data points across telecom subscriber counts, tele-density, gross enrollment ratios, digital transaction volumes, and electricity consumption. The datasets span from 2008 to 2023, giving us over a decade of India's digital transformation story.

We organized this into four research questions, which you can see on screen. We'll walk through each one.

Before we dive in — a quick note on methodology. We used a dual-database architecture: **SQLite** for our structured relational data — tele-density, education, transactions — and **MongoDB** for the provider-level telecom subscription data, which has 58,000 documents with significant sparsity in wireline and VLR fields. This split wasn't arbitrary — it directly reflects what we learned in class about when relational schemas work well versus when document stores are the better fit. The provider-level data has 47 different provider names that we normalized down to 17, and about 56% missing values in the wireline column — a textbook case for MongoDB's sparse document representation.

*(pause, then click to Page 1)*

---

## PAGE 1: Telecom Transformation (click "Telecom Transformation") — SAUMYA

This page addresses our first research question: was the growth in wireless subscribers smooth, or were there identifiable structural breaks?

**[Point to the wireless subscriber growth chart]**

This is the national wireless subscriber count from 2013 to 2023. You can see it grew from about 870 million to 1.17 billion. But the growth wasn't uniform. We applied the **Bai-Perron structural break detection algorithm**. Let me explain what this does.

A structural break is a point in a time series where the statistical properties — the mean, the trend slope, the variance — fundamentally change. You could eyeball this chart and guess where the breaks are, but that's subjective. The Bai-Perron algorithm does this rigorously. It works by fitting a piecewise linear model to the data and searching over all possible break locations to find the split points that minimize the total residual sum of squares. Crucially, it's **endogenous** — we don't tell it where to look. We just say "find 2 breaks" and it searches the entire series. We used the `ruptures` library's Binary Segmentation implementation with an L2 cost model and a minimum segment size of 12 months.

It found two. The first, in **January 2011**, marks the peak of India's initial mobile explosion — the feature phone era where subscriber counts were growing at a CAGR of over 47%. The second break lands in **June 2016** — three months before Jio officially launched in September 2016. The algorithm is picking up the market anticipation and early disruption.

**[Point to the Chow test table]**

We validated both breaks with **Chow tests**. The Chow test is a confirmatory test for structural breaks. Here's how it works: you fit a linear trend model — subscribers as a function of time — on the full series. Then you split the series at the proposed break point and fit separate trend models on each half. If the break is real, two separate models should fit the data much better than a single model. The F-statistic measures exactly this — it compares the pooled residual sum of squares against the sum of the two sub-period residuals. Under the null hypothesis of no break, this follows an F-distribution, giving us a p-value.

Both breaks are statistically significant — the F-statistics are 249 and 42, with p-values well below 0.01. These are not random fluctuations; these are genuine regime changes in India's telecom growth trajectory.

**[Point to the CAGR table]**

The **CAGR — Compound Annual Growth Rate** — tells the story quantitatively. CAGR smooths out volatility by computing the constant annual rate that would take you from the start value to the end value: it's (end/start)^(1/years) minus 1. Pre-2011: 47% annual growth. Between 2011 and Jio's entry: about 5.6%. Post-Jio: 2.8% in raw subscriber numbers — but this is misleading if you only look at quantity. The quality of access changed dramatically. Jio brought 4G data to hundreds of millions who previously had voice-only connections.

**[Scroll down to digital transactions section]**

Now look at the downstream effect on digital payments. Digital transaction volumes grew from nearly zero in 2016 to over 5,000 crore monthly by 2021. The stacked area chart on the right shows the payment composition shift — UPI went from 0.4% of digital payments to 61.5%. Debit cards collapsed from 17% to 3.4%. This is the fastest payment method adoption in any major economy.

You can also see the COVID effect — that red dashed line at March 2020. Contactless payments jumped and never reverted, suggesting a permanent behavioral shift, not a temporary crisis response.

**[Scroll down to provider market share and HHI]**

Finally, the market structure. The stacked area chart shows Jio eating into Airtel, Vodafone, and Idea's market share from 2016 onward. This triggered the Vodafone-Idea merger and forced Airtel to restructure its pricing.

The HHI chart shows the **Herfindahl-Hirschman Index**. HHI measures market concentration by summing the squared market shares of all firms — so if one firm has 100% of the market, HHI is 10,000; if 10 equal firms each have 10%, HHI is 1,000. We computed this from our MongoDB data using an aggregation pipeline: group by state, year, and provider to get each provider's total wireless subscribers, then compute each provider's share of the state total, square those shares, and sum them. The thresholds are standard: below 1,500 is competitive, above 2,500 is highly concentrated.

Before Jio, the market was highly concentrated, above 2,500. After Jio's entry, HHI dropped sharply — the market moved from highly concentrated to moderately concentrated within two years. This is the fastest telecom market disruption in Indian history.

*(pause, then click to Page 2)*

---

## PAGE 2: Connectivity & Education (click "Connectivity & Education") — VATSL

Research question 2: does telecom access translate into higher education enrollment?

**[Point to the scatter plot]**

This scatter plot shows every state-year observation — tele-density on the x-axis, GER on the y-axis. The upward slope is clear: states with higher connectivity tend to have higher enrollment. But we have to be careful — this could be driven by wealth. Rich states have both more towers and more universities.

**[Point to the regression table]**

To control for this, we ran a **lagged two-way fixed effects panel regression**. We used tele-density at year t-1 to predict GER at year t — this is important because if we used contemporaneous data, we couldn't say anything about direction. By lagging, we're testing whether last year's connectivity predicts this year's enrollment.

The coefficients are positive for Total GER and Female GER — the direction is consistent with our hypothesis. However, **none reach statistical significance**. All p-values are above 0.10.

**[Point to the coefficient bar chart with grey bars]**

You can see the grey bars here — that means not significant at the 5% level. This is almost certainly a **statistical power problem**. With only 17 states and 8 time periods, once we absorb state fixed effects and year fixed effects, there's very little within-state variation left to detect what is likely a small but real effect. We treat this as suggestive but inconclusive evidence — the direction is right, but the data is too short to confirm it statistically.

*(pause, then click to Page 3)*

---

## PAGE 3: Digital Payments (click "Digital Payments") — SAUMYA

Research question 3: did wireless growth actually cause digital transaction growth, or are they just correlated?

**[Point to the normalized overlay chart]**

First, the corroboration evidence. We used a **normalized time-series overlay** to visually compare three indicators that come from completely different sources and have different units. Electricity is in gigawatt-hours, wireless subscribers are raw counts in the hundreds of millions, and digital transactions are in crores of rupees. You can't plot them on the same axis directly. So we applied **min-max normalization** to each series independently — subtract the minimum, divide by the range — which maps every series to a 0-to-1 scale while preserving the shape of growth. This lets us see whether the inflection points and acceleration periods align across the three indicators.

And they do. All three rise together from 2010 onward, with a visible acceleration post-2016. The alignment of these three independent data sources from three different government agencies — CEA, TRAI, and RBI — is strong circumstantial evidence that connectivity expansion and digital economic activity are tightly linked.

**[Point to the Granger causality table]**

But visual alignment isn't a statistical test. We wanted to test causality formally, so we ran a **Granger causality test**. Let me explain what this does.

Granger causality asks a precise question: does knowing the past values of wireless growth improve our prediction of digital transaction growth, beyond what past transaction growth alone tells us? It works by fitting two models. Model A — the restricted model — predicts this month's digital transaction growth rate using only its own past values at lags 1 through 4. Model B — the unrestricted model — uses the same past transaction values *plus* past wireless growth values at the same lags. Then we compare the residual sum of squares of both models with an F-test. If Model B's residuals are significantly smaller, wireless growth contains predictive information that transaction history alone doesn't — and we say wireless growth "Granger-causes" transaction growth.

A few important prerequisites. Both series must be **stationary** — no trend or unit root — otherwise the test is invalid. We converted both to month-over-month percentage changes and confirmed stationarity with the **Augmented Dickey-Fuller test** — both series had p-values below 0.001, well stationary. We used max lag of 4 months rather than 6 to preserve degrees of freedom, since we only have about 50 observations after differencing.

The result: **no significant Granger causality at any lag from 1 to 4 months**. The F-statistics are tiny — between 0.01 and 0.31 — and all p-values are above 0.70. Adding wireless growth history does not improve our prediction of transaction growth at all.

This is actually a meaningful finding, not a failure. It tells us that the relationship between telecom and payments doesn't operate at the monthly frequency. Infrastructure rollout effects on payment behavior play out over quarters or years, not months — a tower built in June doesn't show up as UPI transactions in July. The corroboration evidence from the overlay chart supports the long-run relationship, even though the short-run Granger test doesn't capture it. And reporting a null result honestly is stronger than forcing a false positive.

*(pause, then click to Page 4)*

---

## PAGE 4: The Digital Divide (click "The Digital Divide") — VATSL

This is arguably the most policy-relevant part of our analysis. Research question 4: which states are being left behind?

**[Point to the Louvain network graph]**

We built a state-similarity graph. Each node is a state. Edges are weighted by cosine similarity of four features: mean tele-density, tele-density growth slope, mean GER, and GER growth slope. We then ran **Louvain community detection** — an algorithm from our Social Network Analysis module — to identify natural groupings.

It found three communities:
- **Blue (Community 0)**: the digital leaders — Kerala, Tamil Nadu, Karnataka, Maharashtra, Delhi. High tele-density, high GER.
- **Orange (Community 1)**: the digital laggards — Bihar, Assam, UP, West Bengal, Madhya Pradesh, Odisha. Low on both dimensions.
- **Green (Community 2)**: a mid-tier group — Haryana, Punjab, J&K. Between the two extremes.

This three-way split is more nuanced than K-means, which only found 2 clusters. Louvain captures the Punjab-Haryana-J&K mid-tier that K-means lumped into the mainstream.

**[Point to community profiles table]**

The numbers are stark. The leader community has a mean tele-density of 129 — nearly 1.8 times the laggard community's 72. The GER gap is similarly large: 36 versus 20. Digital and educational exclusion are spatially co-located.

**[Point to the gap analysis table]**

And here's the finding that should concern policymakers. We computed how many years it would take each laggard state, at its current growth rate, to reach the leader community's mean tele-density.

Bihar: **135 years**. Rajasthan: over 100 years. These aren't merely behind — they're structurally diverging. Without targeted policy intervention — subsidized infrastructure, mandatory rollout obligations, direct device and data subsidies — this digital divide will not close within any planning horizon.

*(pause, then click to Page 5)*

---

## PAGE 5: State Explorer (click "State Explorer") — VATSL

This is our interactive exploration tool. Let me demonstrate.

**[Select "Kerala" from the dropdown]**

Here's Kerala — one of our digital leaders. Tele-density consistently above 100, strong wireless subscriber growth, GER around 35-40.

**[Toggle "Compare two states", select "Bihar" as State 2]**

Now let's compare with Bihar — our most extreme laggard. The tele-density gap is immediately visible. Bihar's line is consistently 50-60 points below Kerala's. The wireless subscriber chart shows Bihar has massive raw numbers — it's a large state — but per-capita connectivity is far lower. And the GER chart confirms the education gap.

This is exactly the kind of state-level drill-down that makes the digital divide tangible rather than abstract.

*(pause)*

---

## LLM DEMO (point to the sidebar) — SAUMYA

Finally, our bonus component. We built an AI-powered query interface using the Claude API through **LangChain**.

Let me explain the architecture briefly. We use LangChain's `ChatAnthropic` wrapper to send a natural language question to Claude along with a **system prompt** that contains the full schema of both our databases — all six SQLite tables with their column names and types, and the MongoDB collection structure. The system prompt instructs Claude to respond with *only* a Python code block that assigns its output to a variable called `result`. When we receive the response, we extract the code from the markdown fence using a regex, and then execute it dynamically in a Python namespace that has pandas, numpy, sqlite3, pymongo, and plotly pre-loaded, along with our database connection paths. The `result` variable is then displayed — if it's a DataFrame we show a table, if it's a Plotly figure we render the chart, otherwise we print it as text.

This is intentionally a flexible, open-ended approach — the LLM can write any query it wants against our actual data. It's not production-safe, but for a locally hosted one-time demo, it shows the power of combining an LLM with structured databases.

**[Enter API key if not already entered]**

**[Click "Top 5 states by tele-density in 2021"]**

Watch what happens — the system sends my natural language question to Claude, which generates Python code, executes it against our actual SQLite and MongoDB databases, and returns the result. You can see the DataFrame here.

**[Click "Show generated code"]**

And this expander shows you exactly what code was generated and run. Full transparency — you can see the SQL query, the pandas operations, everything. The prof or anyone in the audience can verify that the code is doing what we claim.

**[Type a custom query: "What was the average HHI before and after 2016?"]**

I can also ask arbitrary questions. This one compares market concentration before and after Jio's entry... and there's the result.

This interface can query both our SQL and MongoDB databases, generate charts, compute statistics — anything you'd do in a Jupyter notebook, but through natural language.

---

## CLOSING — SAUMYA

To summarize our key findings:

1. India's wireless growth had **two structural breaks** — 2011 and 2016 — both statistically confirmed. Jio didn't just add subscribers; it restructured the entire market.

2. Tele-density and education enrollment move in the same direction, but our panel is too short to confirm causality. The evidence is **suggestive but inconclusive**.

3. Wireless growth and digital payments are **tightly linked in the long run**, even though monthly Granger causality doesn't reach significance. The payment composition shift — from debit cards to UPI — is the most dramatic in any major economy.

4. The digital divide is **not closing**. Bihar would need over a century at current growth rates to reach Kerala's connectivity level. This demands targeted policy intervention.

We used both SQL and MongoDB to demonstrate relational and document-store paradigms. We applied structural break detection, panel regression, Granger causality, Louvain community detection, and cross-correlation analysis — drawing directly from techniques covered in this course.

Thank you. Happy to take questions.

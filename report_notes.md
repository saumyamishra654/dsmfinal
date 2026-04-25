# DSM Project — Report Notes

> Running log of findings, decisions, and methodology notes. Each section maps to a completed task.
> Expand these bullets into full paragraphs for the final report.

---

## 1. Dataset Overview (EDA)

### area-wise tele density.csv
- 2,100 rows · 21 unique telecom circles · years 2013–2023
- **Tele-density** = phones per 100 people (not per area)
- Year encoded as `"Financial Year (Apr - Mar), 2023"`, month as `"November, 2023"` → need parsing
- No missing values in the density column

### education-enrolment.csv
- 3,231 rows · 36 unique states/UTs · years 2012–2021
- Columns: state, year, gender (Male/Female/Total), category (All Categories / Scheduled Caste / Scheduled Tribe / OBC / Muslim Minority)
- **270 missing GER values** — concentrated in UTs (Lakshadweep, Dadra & NH, etc.) and smaller NE states
- Year encoded as `"Financial Year (Apr - Mar), 2021"` → same FY format as tele density

### digital transactions.csv
- 64 rows · national level only (no state breakdown)
- Year encoded as `"Calendar Year (Jan - Dec), 2022"`, month as `"July, 2022"` → CY format (different from FY above)
- Three value columns: total digital txn, BHIM/UPI txn, debit card txn — all in crores of INR (scaling factor 10M)
- No missing values

### sector-wise electricity consumption.csv
- 158 rows · sectors: Industry, Agriculture, Domestic, Commercial, Railways, Others
- Years 1970–2023 (FY format)
- `pct_growth` column has some missing values — interpolate for visualisation, exclude from statistical tests

### wired, wireless telephone.csv
- 1,694 rows · 22 unique service areas · years 2013–2023 (CY format)
- **Differs from tele density:** uses `"Uttar Pradesh (East)"` and `"Uttar Pradesh (West)"` as separate circles
- Values in millions of subscribers; `pct_share` = wireless share of total phones

---

## 2. Data Cleaning Decisions

### 2.1 Telecom Circle → State Mapping

**Problem:** The telecom datasets use TRAI "circle" names, not state names. The education dataset uses official state names. They don't match 1:1.

**Specific mismatches found:**
- `Kolkata` → a metro circle inside West Bengal (not a state)
- `Mumbai` → a metro circle inside Maharashtra (not a state)
- `North East` → one circle covering 7 NE states (Arunachal Pradesh, Manipur, Meghalaya, Mizoram, Nagaland, Sikkim, Tripura) — cannot attribute to any individual state
- `Jammu & Kashmir` → name mismatch; education uses `"Jammu And Kashmir"`

**States in education data with NO direct circle** (folded into larger TRAI circles):
- Chhattisgarh → inside Madhya Pradesh circle
- Jharkhand → inside Bihar circle
- Uttarakhand → inside Uttar Pradesh circle
- Goa → inside Maharashtra circle
- Telangana → inside Andhra Pradesh circle (pre-2014 split not reflected in TRAI circles)
- All 7 NE states → inside North East circle
- All UTs (Chandigarh, Puducherry, Lakshadweep, A&N Islands, Dadra & NH, Ladakh) → inside adjacent state circles

**Decision:** Use only the **17 major states** where the telecom circle name directly maps to a state name (after J&K normalisation). These are: Andhra Pradesh, Assam, Bihar, Delhi, Gujarat, Haryana, Himachal Pradesh, Jammu & Kashmir, Karnataka, Kerala, Madhya Pradesh, Maharashtra, Odisha, Punjab, Rajasthan, Tamil Nadu, Uttar Pradesh, West Bengal.

**Rationale:** Metro sub-circles (Kolkata, Mumbai) would double-count their states. North East cannot be disaggregated. Carved-out states (Chhattisgarh, Jharkhand, Uttarakhand) share a TRAI circle with their parent state, so attributing the parent's tele-density to them would be misleading. Document as a limitation.

**Note on Kolkata/Mumbai:** These are excluded from the panel rather than merged with WB/Maharashtra, to avoid inflating tele-density for those states (WB circle + Kolkata circle would average out distortedly).

### 2.2 Date Parsing

- FY string `"Financial Year (Apr - Mar), 2023"` → extract year as integer (2023)
- Month string `"November, 2023"` → extract month as integer (11)
- CY string `"Calendar Year (Jan - Dec), 2022"` → extract year as integer (2022)
- Construct `DATE` column as `YYYY-MM-01` for time-series ordering
- FY vs CY alignment: FY 2023 month April = CY 2022-04 (April is the start of the Indian financial year, so FY 2023 spans Apr 2022 – Mar 2023)

### 2.3 Uttar Pradesh Split (wired_wireless only)
- `wired, wireless telephone.csv` splits UP into `"Uttar Pradesh (East)"` and `"Uttar Pradesh (West)"`
- `area-wise tele density.csv` uses a single `"Uttar Pradesh"` entry
- **Decision:** In `load_sqlite.py`, aggregate UP East + UP West by summing wireline/wireless millions and recomputing pct_share before inserting into `wired_wireless` table. Both rows FK to the same `state_id` for Uttar Pradesh.

### 2.4 Missing GER Values
- 270 NULLs, mostly in UTs and smaller NE states
- These states also lack telecom circle data → they would be excluded from the panel anyway
- Store as SQL `NULL`; exclude from regression; note in report

---

## 3. src/data/clean.py

- Single module of pure utility functions; no DB imports, importable from both `load_sqlite.py` and `load_mongo.py`
- `parse_year(raw)` / `parse_month(raw)` — regex-based extraction from TRAI string formats
- `fy_month_to_cy(fy_year, month)` — converts Financial Year label + month to Calendar Year (months Apr–Dec shift back one year)
- `make_date(year, month)` → `"YYYY-MM-01"` string for SQL `DATE` column
- `CIRCLE_TO_STATE` dict — 19 entries mapping TRAI circle names to canonical state names; Kolkata, Mumbai, North East explicitly omitted
- `PANEL_STATES` list — 17 canonical state names available for the panel regression
- Five `clean_*()` functions, one per dataset, returning clean DataFrames ready for DB insertion
- Sanity check (row counts after filtering): tele_density 2100→1800, wired_wireless 1694→1386, education 3231 (all rows kept, NULLs preserved), digital_transactions 64, electricity 158
- `pct_growth` in electricity: 60 missing values confirmed — will interpolate for viz, exclude from stats
- Cleaned DataFrames written to `cleaned_datasets/*.pkl` (type-preserving; gitignored like the DB file); `load_sqlite.py` and `load_mongo.py` both read from here so both DBs are guaranteed to use identical cleaned data

---

## 4. src/data/load_sqlite.py

### Schema Design

**`states`** — dimension / lookup table
- `state_id` PK, `state_name` UNIQUE
- 36 rows: India has 28 states + 8 Union Territories = 36 administrative divisions. The education dataset covers all of them, so the states table holds all 36. Most UTs lack GER data and have no telecom circle, so they are excluded from the panel analysis but legitimately present in the dimension table.
- All other state-level tables FK to this table, enforcing referential integrity.

**`tele_density`** — monthly tele-density per circle (post-mapping: per state)
- `state_id` FK, `year` INT, `month` INT, `date` DATE, `tele_density` REAL
- 1,800 rows (from 2,100 raw; 300 excluded because their circle — Kolkata, Mumbai, North East — had no clean state mapping)
- Composite index on `(state_id, date)` for time-series queries

**`wired_wireless`** — monthly wireline/wireless subscriber counts per state
- `state_id` FK, `year` INT, `month` INT, `date` DATE, `wireline_millions` REAL, `wireless_millions` REAL, `pct_share` REAL
- 1,386 rows (from 1,694 raw; UP East + UP West collapsed into one UP row per month; excluded circles dropped)
- Composite index on `(state_id, date)`

**`education_ger`** — annual Gross Enrolment Ratio per state × gender × category
- `state_id` FK, `year` INT, `gender` TEXT, `category` TEXT, `ger` REAL (nullable)
- 3,231 rows; 270 NULL GER values stored as SQL NULL
- Gender values: Male / Female / Total
- Category values: All Categories / Scheduled Caste / Scheduled Tribe / OBC / Muslim Minority
- Index on `(state_id, year)` for panel JOIN with tele_density

**`digital_transactions`** — national monthly digital payment volumes
- No state FK (national-level only — a known limitation for state-level analysis)
- `year` INT, `month` INT, `date` DATE, `digital_txn_crores` REAL, `bhim_txn_crores` REAL, `debit_card_crores` REAL
- 64 rows; no missing values

**`electricity_consumption`** — annual sector-wise electricity consumption
- No state FK (national aggregate)
- `year` INT, `sector` TEXT, `additional_info` TEXT, `energy_gwh` REAL, `pct_consumption` REAL, `pct_growth` REAL (nullable)
- 158 rows; 60 NULL `pct_growth` values (early years pre-1980 had no growth rate recorded)

### Normalization — Why state_id Instead of Storing state_name Directly
- The `states` table is a **3NF normalization** decision: `state_name` is stored exactly once; all other tables reference it via the integer `state_id` FK
- **Data redundancy eliminated:** without this, the string `"Andaman And Nicobar Islands"` would repeat thousands of times across `tele_density`, `wired_wireless`, and `education_ger`
- **Update anomaly prevented:** if a state name needed correction, you'd change one row in `states` instead of hunting through thousands of rows across multiple tables
- **Referential integrity enforced:** a FK constraint makes it physically impossible to insert a row in `tele_density` with a `state_id` that doesn't exist in `states` — a typo in a direct state_name column would silently create a dangling row
- Technique: **surrogate key normalization** — synthetic integer PK used as the join key instead of the natural key (`state_name`)

### Implementation Notes
- DB is dropped and recreated on every run (idempotent)
- `PRAGMA foreign_keys = ON` enforced at connection time
- NaN values converted to Python `None` before insert → stored as SQL NULL

---

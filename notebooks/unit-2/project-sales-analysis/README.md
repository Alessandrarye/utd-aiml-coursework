# Project 1 — AAL Australian Apparel Sales Analysis (Q4 2020)

**Deliverable:** One JupyterLab notebook, e.g. `AAL_Sales_Analysis.ipynb`
**Input file:** `AusApparalSales4thQrt2020.csv`
**Columns you'll find:** `Date`, `Time`, `State`, `Group`, `Unit`, `Sales`

Structure the notebook in four sections matching the rubric: Wrangling → Analysis → Visualization → Report. Use a Markdown cell before each code cell explaining what you're doing and *why* — the graders (and your narration format) reward this.

---

## Section 0 — Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
%matplotlib inline

df = pd.read_csv("AusApparalSales4thQrt2020.csv")
df.head()
```

---

## Section 1 — Data Wrangling

### Step 1.1 — Inspect the data
```python
df.info()
df.describe()
df.shape
```

### Step 1.2 — Missing / incorrect entries (required: isna() and notna())
```python
df.isna().sum()          # count of nulls per column
df.notna().sum()         # count of valid entries per column
df.isna().any()          # quick boolean check
```
This dataset is usually complete, but **state your treatment recommendation anyway** in a Markdown cell:
- If nulls existed in `Sales`/`Unit` (numeric): impute with the **median per State+Group** (robust to skew) rather than dropping, since Q4 data is small and every row matters.
- If nulls existed in categorical columns (`State`, `Group`): drop those rows — you can't reliably guess a category.

### Step 1.3 — Check for *incorrect* entries (the hidden gotcha)
Categorical columns in this file often carry **leading/trailing whitespace** (e.g. `" Morning "`). Clean them or your groupbys will silently split categories:
```python
for col in ["Time", "State", "Group"]:
    df[col] = df[col].str.strip()

df["Time"].unique(), df["State"].unique(), df["Group"].unique()
```
Also convert `Date`:
```python
df["Date"] = pd.to_datetime(df["Date"])
```
Sanity checks: no negative `Unit` or `Sales`, dates all fall in Oct–Dec 2020:
```python
(df[["Unit", "Sales"]] < 0).any()
df["Date"].min(), df["Date"].max()
```

### Step 1.4 — Normalization (required — the brief says normalization is preferred)
Use **Min-Max normalization** (scales to [0, 1]). Explain in Markdown: Sales and Unit are on very different scales, and normalization preserves the shape of the distribution while making the two comparable. Standardization (z-scores) is better when data is Gaussian and you care about distance-based models; here we just want comparable magnitudes for reporting/visualization.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[["Unit_norm", "Sales_norm"]] = scaler.fit_transform(df[["Unit", "Sales"]])
df[["Unit", "Unit_norm", "Sales", "Sales_norm"]].head()
```
(Keep the original columns — you'll report real dollar figures, and use normalized ones for comparison plots.)

### Step 1.5 — GroupBy discussion (required insight)
Demonstrate both uses, then recommend:
```python
# Chunking: split data into groups and aggregate
df.groupby("State")["Sales"].sum().sort_values(ascending=False)
df.groupby(["State", "Group"])["Sales"].agg(["sum", "mean"])
```
**Recommendation to write:** `groupby()` is best used here for **data chunking (split-apply-combine)** — the business questions are all "revenue *by* state / group / time," which is exactly what chunked aggregation answers. Merging (`pd.merge`) is for combining separate tables; we have a single table, so merging adds nothing here.

---

## Section 2 — Data Analysis

### Step 2.1 — Descriptive statistics on Sales and Unit
```python
for col in ["Sales", "Unit"]:
    print(f"--- {col} ---")
    print("Mean:  ", df[col].mean())
    print("Median:", df[col].median())
    print("Mode:  ", df[col].mode()[0])
    print("Std:   ", df[col].std())
```
Or the compact version: `df[["Sales", "Unit"]].describe()` plus `stats.mode()` from SciPy (this checks the "use SciPy" box):
```python
stats.mode(df["Sales"], keepdims=True)
```

### Step 2.2 — Highest and lowest performers
```python
group_sales = df.groupby("Group")["Sales"].sum().sort_values(ascending=False)
state_sales = df.groupby("State")["Sales"].sum().sort_values(ascending=False)
print(group_sales)   # highest = first row, lowest = last row
print(state_sales)
```
Write one Markdown cell interpreting each. (Typical result: VIC leads, WA trails; Men/Women lead, Seniors trail — but report what *your* numbers say.)

### Step 2.3 — Weekly, monthly, quarterly reports
```python
ts = df.set_index("Date")

weekly    = ts.resample("W")["Sales"].sum()
monthly   = ts.resample("ME")["Sales"].sum()   # use "M" on older pandas
quarterly = ts.resample("QE")["Sales"].sum()   # use "Q" on older pandas

weekly, monthly, quarterly
```
Present each as a small table (and reuse them for plots in Section 3).

---

## Section 3 — Data Visualization (the dashboard)

State your library recommendation in Markdown first: **Seaborn**, because it's built for statistical plotting (built-in aggregation, confidence intervals, distribution plots), produces publication-quality defaults, and layers on Matplotlib so you keep full control. Then build the dashboard — a 2×2 or 3×2 `plt.subplots` grid works well and literally looks like a dashboard.

### Step 3.1 — State-wise sales by demographic group
```python
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="State", y="Sales", hue="Group", estimator=sum, errorbar=None)
plt.title("State-wise Sales by Demographic Group")
plt.show()
```

### Step 3.2 — Group-wise sales across states (the inverse view)
```python
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Group", y="Sales", hue="State", estimator=sum, errorbar=None)
plt.title("Group-wise Sales across States")
plt.show()
```

### Step 3.3 — Time-of-day analysis (peak / off-peak)
```python
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Time", y="Sales", estimator=sum, errorbar=None,
            order=["Morning", "Afternoon", "Evening"])
plt.title("Sales by Time of Day")
plt.show()
```
Interpret: name the peak window and tie it to the S&M ask (schedule staff, push Next-Best-Offer promos into the peak window; run hyper-personalized offers to lift the off-peak window).

### Step 3.4 — Daily, weekly, monthly, quarterly charts (all four are required)
```python
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

ts.resample("D")["Sales"].sum().plot(ax=axes[0,0], title="Daily Sales")
weekly.plot(ax=axes[0,1], marker="o", title="Weekly Sales")
monthly.plot(ax=axes[1,0], kind="bar", title="Monthly Sales")
quarterly.plot(ax=axes[1,1], kind="bar", title="Quarterly Sales")

plt.tight_layout()
plt.show()
```

### Step 3.5 — Required statistical plots
Box plot (required for descriptive stats):
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x="Group", y="Sales", ax=axes[0])
sns.boxplot(data=df, x="State", y="Sales", ax=axes[1])
plt.tight_layout(); plt.show()
```
Seaborn distribution plot (required):
```python
sns.displot(df["Sales"], kde=True)
plt.title("Distribution of Sales")
plt.show()
```

---

## Section 4 — Report Generation

This is mostly Markdown discipline inside the same notebook:

1. **Title cell** — project name, your name, date.
2. **Executive summary** — 4–6 bullets: top state, bottom state, top group, peak time, trend across the quarter.
3. **Section headers** (`##`) before each of the four phases.
4. **Interpretation cell after every chart** — one or two sentences each.
5. **Recommendations cell** at the end, aimed at the S&M head, e.g.:
   - Concentrate expansion capital in the top-revenue states; they've proven demand.
   - For low-revenue states (likely WA/NT/TAS): localized promotions, group-targeted campaigns for the weakest demographic, and off-peak discount programs.
   - Staff and promote around the identified peak time window; use hyper-personalization to lift off-peak.
6. **Restart & Run All** before submitting so outputs are clean and sequential.

**Submission checklist:** isna/notna shown ✅ · treatment recommendation written ✅ · normalization executed and shown ✅ · groupby insight + recommendation ✅ · mean/median/mode/std ✅ · highest/lowest identified ✅ · weekly/monthly/quarterly tables ✅ · daily/weekly/monthly/quarterly charts ✅ · box plot ✅ · seaborn distribution plot ✅ · library recommendation ✅ · Markdown throughout ✅

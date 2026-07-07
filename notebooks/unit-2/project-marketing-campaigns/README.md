# Project 2 — Marketing Campaigns: EDA + Hypothesis Testing

**Deliverable:** One JupyterLab notebook, e.g. `Marketing_Campaign_Analysis.ipynb`
**Input file:** the CSV from the Google Drive link (commonly `marketing_data.csv`)
**Typical columns:** `ID, Year_Birth, Education, Marital_Status, Income, Kidhome, Teenhome, Dt_Customer, Recency, MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds, NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth, AcceptedCmp1–5, Response, Complain, Country`

Map columns to the 4 Ps in an early Markdown cell (People = demographics, Product = the `Mnt*` spend columns, Place = the purchase-channel columns, Promotion = the `AcceptedCmp*`/`Response` columns). It shows you understood the framing.

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

df = pd.read_csv("marketing_data.csv")
df.head()
```

---

## Step 1 — Verify import of Dt_Customer and Income

Two known traps in this dataset:

**Trap A — column name whitespace.** `Income` is often stored as `" Income "`:
```python
df.columns = df.columns.str.strip()
```

**Trap B — Income is a string with $ and commas** (`"$84,835.00"`):
```python
df["Income"] = (df["Income"]
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .astype(float))
```

**Dt_Customer is a string** — parse it (check the format in the file; it's usually month/day/year):
```python
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%m/%d/%y")
```
Confirm with `df.info()` — Income should now be float64, Dt_Customer datetime64. Screenshot-worthy before/after.

---

## Step 2 — Clean categories, then impute missing Income

### 2.1 Inspect the messy categories
```python
df["Education"].value_counts()
df["Marital_Status"].value_counts()
```
`Marital_Status` contains junk levels: `Alone`, `Absurd`, `YOLO`. Consolidate:
```python
df["Marital_Status"] = df["Marital_Status"].replace({
    "Alone": "Single",
    "Absurd": "Single",
    "YOLO": "Single",
})
```
Optionally simplify Education too (e.g. `"2n Cycle" → "Master"` is a defensible mapping — 2n Cycle is the Bologna-system second cycle). Whatever you choose, justify it in Markdown.

### 2.2 Impute Income by Education + Marital_Status group mean
The brief tells you the imputation logic explicitly — use it:
```python
df["Income"].isna().sum()   # ~24 missing

df["Income"] = df["Income"].fillna(
    df.groupby(["Education", "Marital_Status"])["Income"].transform("mean")
)

df["Income"].isna().sum()   # should be 0
```
`transform("mean")` broadcasts each group's mean back to the row level — a clean one-liner worth explaining in Markdown.

---

## Step 3 — Feature engineering

```python
# Total children
df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

# Age — use a fixed reference year (dataset is ~2014-era) and justify it
df["Age"] = 2014 - df["Year_Birth"]

# Total spending across all product categories
mnt_cols = ["MntWines", "MntFruits", "MntMeatProducts",
            "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
df["Total_Spending"] = df[mnt_cols].sum(axis=1)

# Total purchases across the THREE channels (deals is a discount flag, not a channel)
df["Total_Purchases"] = (df["NumWebPurchases"]
                         + df["NumCatalogPurchases"]
                         + df["NumStorePurchases"])
```

---

## Step 4 — Distributions, box plots, outlier treatment

```python
num_cols = ["Income", "Age", "Total_Spending", "Total_Purchases"]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, ax=axes[0, i])
    sns.boxplot(y=df[col], ax=axes[1, i])
plt.tight_layout(); plt.show()
```

You will find two famous outliers:
- **Age ≈ 121+** — three customers born in 1893/1899/1900. Impossible; drop them.
- **Income = 666,666** — an obvious sentinel/typo. Drop or cap it.

Two defensible treatments (pick one, justify it):
```python
# Option A: targeted drop of impossible values
df = df[(df["Age"] < 100) & (df["Income"] < 600000)].copy()

# Option B: IQR capping (winsorize) — keeps all rows
def iqr_cap(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return s.clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
```
Re-plot the box plots after treatment to show the before/after.

---

## Step 5 — Encoding

Two kinds required — explain the distinction in Markdown:

**Ordinal** (Education has a natural order):
```python
edu_order = {"Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4}
df["Education_Encoded"] = df["Education"].map(edu_order)
```

**One-hot** (Marital_Status and Country have no order):
```python
df = pd.get_dummies(df, columns=["Marital_Status", "Country"],
                    prefix=["Marital", "Country"], dtype=int)
```
Tip: do the encoding on a copy (`df_enc`) if you want to keep the readable categorical columns for the visualization section.

---

## Step 6 — Correlation heatmap

```python
plt.figure(figsize=(14, 10))
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap")
plt.show()
```
Call out 3–4 notable pairs in Markdown (e.g. Income ↔ Total_Spending strongly positive; Kidhome ↔ spending negative; NumWebVisitsMonth ↔ Income negative).

---

## Step 7 — Hypothesis tests (the core of the grade)

For each: state H0/H1 in Markdown, run the test, report the p-value, conclude at α = 0.05.

### H1 — Older customers prefer in-store shopping
Compare age's relationship with store vs. web purchases:
```python
r_store, p_store = stats.pearsonr(df["Age"], df["NumStorePurchases"])
r_web,   p_web   = stats.pearsonr(df["Age"], df["NumWebPurchases"])
```
Also useful: split at the median age and t-test store-purchase *share*:
```python
df["Store_Share"] = df["NumStorePurchases"] / df["Total_Purchases"].replace(0, np.nan)
older   = df[df["Age"] >  df["Age"].median()]["Store_Share"].dropna()
younger = df[df["Age"] <= df["Age"].median()]["Store_Share"].dropna()
stats.ttest_ind(older, younger, equal_var=False)
```
Typical finding: age correlates weakly with *both* channels — the "seniors avoid tech" assumption is not strongly supported. Report what your numbers show.

### H2 — Customers with children prefer online shopping
```python
with_kids = df[df["Total_Children"] > 0]
no_kids   = df[df["Total_Children"] == 0]

# One-sided: parents buy MORE online?
stats.ttest_ind(with_kids["NumWebPurchases"], no_kids["NumWebPurchases"],
                equal_var=False, alternative="greater")
```
Also compare the *web share* of purchases, since parents spend less overall — raw counts can mislead. (Typical finding: the hypothesis is rejected in raw counts; discuss the nuance.)

### H3 — Store sales cannibalized by other channels
Cannibalization would show as a *negative* correlation between store purchases and web/catalog purchases:
```python
stats.pearsonr(df["NumStorePurchases"], df["NumWebPurchases"])
stats.pearsonr(df["NumStorePurchases"], df["NumCatalogPurchases"])
```
Typical finding: correlations are *positive* (heavy shoppers shop everywhere) → no evidence of cannibalization. Note the confounder: overall engagement drives all channels up together.

### H4 — Does the US significantly outperform the rest of the world in total purchases?
```python
us   = df[df["Country_US"] == 1]["Total_Purchases"]   # column name depends on your one-hot prefix
rest = df[df["Country_US"] == 0]["Total_Purchases"]

stats.ttest_ind(us, rest, equal_var=False, alternative="greater")
```
Typical finding: not significant → fail to reject H0. Also worth noting the sample imbalance (US is a small slice of this dataset).

---

## Step 8 — Targeted visualizations

**8a. Top / bottom products by revenue**
```python
product_rev = df[mnt_cols].sum().sort_values(ascending=False)
sns.barplot(x=product_rev.values, y=product_rev.index)
plt.title("Revenue by Product Category")
```
(Wines dominate; fruits/sweets trail — report your actual numbers.)

**8b. Age vs. last-campaign acceptance** (`Response` is the last campaign)
```python
sns.boxplot(data=df, x="Response", y="Age")
stats.pointbiserialr(df["Response"], df["Age"])   # quantify it
```

**8c. Country with the most last-campaign acceptances**
If you one-hot encoded Country, keep a pre-encoding copy, or reconstruct:
```python
country_resp = df_original.groupby("Country")["Response"].sum().sort_values(ascending=False)
sns.barplot(x=country_resp.index, y=country_resp.values)
```
Mention both absolute counts (Spain usually wins — it's the biggest segment) and acceptance *rate*, which tells a different story.

**8d. Children at home vs. total expenditure**
```python
sns.boxplot(data=df, x="Total_Children", y="Total_Spending")
```
Clear pattern: spending drops sharply as children increase — a strong, reportable insight.

**8e. Education of complainers (last 2 years)**
```python
complainers = df_original[df_original["Complain"] == 1]
sns.countplot(data=complainers, y="Education",
              order=complainers["Education"].value_counts().index)
```
Note the tiny base (≈20 complaints) so proportions, not raw counts, are the honest read.

---

## Step 9 — Wrap-up

End the notebook with a Markdown **Conclusions & Recommendations** cell: one line per hypothesis (supported / not supported + p-value), the product mix insight, the children-vs-spending insight, and 2–3 marketing recommendations (e.g., wine-led premium bundles for high-income childless segments; web-channel promotions don't need to be parent-targeted based on H2). Restart & Run All before export.

**Submission checklist:** Dt_Customer + Income verified ✅ · category cleanup shown ✅ · group-mean imputation ✅ · Total_Children / Age / Total_Spending / Total_Purchases created ✅ · hist + box plots with outlier treatment ✅ · ordinal + one-hot encoding ✅ · heatmap ✅ · four hypotheses with named tests and p-values ✅ · five targeted visualizations ✅ · conclusions cell ✅

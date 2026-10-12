# Streamlining the Customer Grievance Process (NLP Project)

NLP analysis of 7,011 retail-banking customer complaints (Jan 1 – Oct 21, 2023): text preprocessing, TF-IDF features, a department classifier for automated triage, VADER sentiment analysis, a transformer comparison, and business recommendations.

## Contents

| Path | What it is |
|---|---|
| `01_complaint_analysis.ipynb` | Main notebook (already executed): data understanding, `preprocessing()`, TF-IDF, department classification, VADER sentiment, topic modeling, business insights |
| `02_transformer_models.ipynb` | Fine-tuned DistilBERT classifier vs. TF-IDF, and RoBERTa sentiment vs. VADER (executed on an RTX 5060 Ti) |
| `data/banking_complaints_2023.xlsx` | Source dataset |
| `figures/` | Every chart, saved as PNG by the notebooks |
| `outputs/split_ids.csv` | Train/test assignment, shared by both notebooks so the models are compared on identical complaints |
| `outputs/tfidf_metrics.json` | TF-IDF model test metrics (read by notebook 02) |
| `outputs/complaints_scored.csv` | Each complaint's VADER scores, severity tier and root-cause topic (no free text) |
| `outputs/transformer_metrics.json`, `outputs/transformer_sentiment.csv` | DistilBERT test metrics and per-complaint RoBERTa sentiment (written by notebook 02) |
| `requirements.txt` | Python dependencies |

## How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace 01_complaint_analysis.ipynb   # ~2 min on CPU
jupyter nbconvert --to notebook --execute --inplace 02_transformer_models.ipynb   # GPU: a few minutes
```

Run both notebooks from this folder, since they use relative paths. Notebook 01 downloads the NLTK `stopwords` and `wordnet` corpora on first run. Notebook 02 downloads `distilbert-base-uncased` and `cardiffnlp/twitter-roberta-base-sentiment-latest` from Hugging Face (about 750 MB). It uses bf16 mixed precision automatically on GPUs that support it.

## Rubric map

| Rubric criterion | Where |
|---|---|
| 1. Data Understanding & Preprocessing | 01 §1 (data types fixed, including ZIP leading zeros; date range; leakage check) and §2 (`preprocessing()`: lowercase → numbers → stopwords → punctuation → lemmatize) |
| 2. Text Feature Engineering | 01 §3: TF-IDF (1–2 grams, fit on train only) → 5,587 × 36,092 sparse matrix used for modeling |
| 3. Complaint Classification | 01 §4: Department as target; 4 models compared with 5-fold CV, grid search, held-out test, confusion matrix, error analysis, confidence-based triage |
| 4. Transformer Modeling & Sentiment | 01 §5 (VADER `SentimentIntensityAnalyzer`, domain-adjusted and interpreted) and 02 (fine-tuned DistilBERT + RoBERTa sentiment) |
| 5. Business Insights & Strategy | 01 §5.5 and §7: how the bank can use the score, six strategies, caveats |

## Key results (notebook 01)

* **Department classifier (TF-IDF + Logistic Regression):** 78.8% test accuracy, macro-F1 0.734 (majority-class baseline: 23.6%). Mortgage F1 0.91. The weakest class, *Others* (F1 0.40), is a legacy label that duplicates CASA.
* **Triage:** auto-routing complaints with confidence ≥ 0.70 handles **~50% of volume at 91% accuracy**.
* **VADER needs domain tuning:** out of the box, it calls 41% of complaints *positive* (because *credit*, *interest* and *please help* score as positive). After neutralizing 56 banking and politeness terms, 80% score negative.
* **Root causes:** *fraud claims & disputed transactions* is the most intense grievance (39% Critical tier) and has the highest payout rate (43%). *Customer-service runaround* is the largest topic (960 complaints).
* **Sentiment → outcomes:** Critical-tier complaints end in monetary relief 25% of the time vs 15% for calm ones. After controlling for length and department, the odds ratio is about 1.23, concentrated in Credit Cards and CASA.

## Key results (notebook 02)

* **Fine-tuned DistilBERT:** macro-F1 **0.767 vs 0.734** for TF-IDF, with accuracy tied (78.9% vs 78.8%). The gain comes from the small *Others* class (F1 0.40 → 0.68) and CASA. TF-IDF stays slightly ahead on Mortgage, Loans and Credit Reports, where keywords are distinctive.
* **RoBERTa sentiment vs VADER:** RoBERTa labels 70% of complaints negative and essentially none positive, which confirms that out-of-the-box VADER's 41% "positive" was a lexicon artifact. But the two methods rank individual complaints very differently (Spearman ρ 0.10–0.22). The VADER severity tiers are therefore a coarse flag, and should be validated against a hand-labeled sample before they drive SLAs.

## Notes

* `Banking Product` maps one-to-one onto `Department`, so it is excluded from the classifier's features (target leakage).
* VADER is applied to the raw text, not the preprocessed text, because preprocessing strips the negations, capitalization and punctuation VADER relies on.

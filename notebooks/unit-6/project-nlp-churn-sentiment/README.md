# NLP Sentiment Analysis + Churn Prediction

UTD AI/ML Bootcamp project. A financial organization wants to improve its churn prediction system by adding customer sentiment, extracted from free-text feedback, to the structured customer data it already models. This project builds an NLP sentiment pipeline over customer feedback, merges the results with the churn dataset on `CustomerId`, and measures whether sentiment improves churn prediction.

## Contents

```
project-nlp-churn-sentiment/
├── nlp_churn_sentiment.ipynb   # the full analysis, organized by Task 1 through Task 7
├── requirements.txt
├── data/
│   ├── README.md               # where the two CSVs go
│   ├── Churn_Modeling.csv      # (add)
│   └── Customer_Feedback.csv   # (add)
└── outputs/                    # created on run: CSVs and figures
    └── figures/
```

## What the notebook does

| Task | Work |
|---|---|
| 1. Load and explore | Shapes, dtypes, missing values, duplicates, rating distribution, feedback length, overlap between the two files' customer IDs |
| 2. Clean and preprocess | Lowercase, expand negative contractions, strip URLs/numbers/punctuation, NLTK tokenization, stopword removal (negations kept), POS-aware WordNet lemmatization |
| 3. Sentiment labels | Star ratings mapped to Negative / Neutral / Positive, plus VADER and TextBlob polarity. Agreement between methods (percent agreement and Cohen's kappa). Optional RoBERTa pre-trained model |
| 4. Vectorize | Stratified train/test split, then Bag-of-Words and TF-IDF (unigrams + bigrams) fit on train only. Optional MiniLM sentence embeddings. Check for identical texts across splits |
| 5. Model and evaluate | Logistic Regression, Linear SVM, Multinomial Naive Bayes, and a Bidirectional LSTM. Accuracy, macro precision/recall/F1, weighted F1, confusion matrix, 5-fold CV |
| 6. Integrate | Out-of-fold sentiment predictions for every feedback row, aggregated per customer, left-joined to the churn data. Churn models (Logistic Regression, Gradient Boosting) compared with and without sentiment features using 5-fold CV ROC-AUC and F1, plus permutation importance |
| 7. Visualize | Sentiment distribution, polarity histograms, VADER vs TextBlob, word clouds per class, correlation heatmap, point-biserial correlations, churn rate by sentiment and by polarity bin, chi-square test |

## Running it

1. Copy `Churn_Modeling.csv` and `Customer_Feedback.csv` into `data/`.
2. Install dependencies if the environment does not already have them:
   ```bash
   pip install -r requirements.txt
   ```
   The NLTK resources (stopwords, WordNet, tokenizer, tagger, VADER lexicon) download automatically on the first run.
3. Run the notebook top to bottom.

### Configuration (top of the notebook)

| Setting | Default | Purpose |
|---|---|---|
| `ID_COL_OVERRIDE`, `TEXT_COL_OVERRIDE`, `RATING_COL_OVERRIDE` | `None` | The feedback file's columns are detected by name. Set these if detection picks the wrong one |
| `LABEL_SOURCE` | `"auto"` | Uses star ratings when present, otherwise VADER. Can force `"rating"`, `"vader"`, `"textblob"`, or `"transformer"` |
| `RUN_LSTM` | `True` | BiLSTM needs TensorFlow; skipped with a message if it is not installed |
| `RUN_SENTENCE_EMBEDDINGS` | `False` | Adds MiniLM embeddings as a third feature set |
| `RUN_TRANSFORMER_SENTIMENT` | `False` | Scores feedback with `cardiffnlp/twitter-roberta-base-sentiment-latest` |

### Dev container notes

* The classical parts run in any Python environment. The LSTM needs TensorFlow, so run the full notebook in the deep learning container.
* For TF 2.21 on the RTX 5060 Ti, the LSTM cell already sets `TF_USE_LEGACY_KERAS=0` before importing TensorFlow and compiles with `jit_compile=False`.
* The repo's `.gitignore` excludes `data`, so the CSVs need a force-add if they should be committed: `git add -f data/*.csv`.

## Outputs

Written to `outputs/` on each run:

* `feedback_with_sentiment.csv`: every feedback row with cleaned text, VADER/TextBlob scores, labels, and the out-of-fold predicted sentiment
* `unified_churn_sentiment.csv`: one row per churn customer with structured and sentiment features combined
* `sentiment_model_comparison.csv`, `churn_model_comparison.csv`: metric tables
* `figures/*.png`: every chart in the notebook

## Results summary

Run on the course data: 19,181 feedback reviews (labels from star ratings) and 10,000 churn customers, 9,957 of whom have feedback.

| Question | Result |
|---|---|
| Label mix | 89% Negative, 2% Neutral, 10% Positive (ratings are mostly 1 star) |
| Best sentiment classifier | Logistic Regression + TF-IDF: accuracy 0.950, macro F1 0.668 (5-fold CV macro F1 0.648) |
| BiLSTM | accuracy 0.924, macro F1 0.612; overfits quickly, early stopping restores the best epoch |
| Hardest class | Neutral (291 examples, F1 about 0.2) |
| Sentiment vs churn | No relationship: every correlation with `Exited` is about 0.01 (p > 0.25); chi-square p = 0.43 |
| Churn model gain from sentiment | ROC-AUC change between -0.001 and +0.003, inside fold-to-fold noise |

**Why sentiment does not predict churn here:** the feedback is public reviews of 47 US banks (locations such as "Magnolia, TX", banks such as Merrick and Chase), assigned row by row to the first 10,000 customer IDs. About 96% of matched customers have a US review location while the churn data places them in France, Spain, or Germany. The pipeline works, but this pairing contains no real sentiment-to-churn signal, which the near-zero gain correctly reflects.

## Method notes

* **Labels:** star ratings are preferred because they are the customer's own judgment. When the labels come from VADER instead, the text classifiers are learning to reproduce VADER, so their scores describe that and not true sentiment accuracy.
* **Negations are kept** during stopword removal, so "not happy" does not turn into "happy".
* **Lexicon scorers use raw text**, since VADER and TextBlob rely on capitalization, punctuation, and negation.
* **No leakage into the merge:** each feedback row's predicted sentiment comes from a model that never trained on that row (5-fold `cross_val_predict`).
* **Customers without feedback** stay in the unified data with neutral sentiment values and `has_feedback = 0`. Correlations in Task 7 use feedback customers only so the filled zeros do not distort them.
* **Timing caveat:** sentiment only adds real predictive value if the feedback was collected before the churn decision. Exit-survey feedback would leak the outcome.

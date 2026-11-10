# Predictive Modeling and Deployment: End-to-End MLOps Pipeline

A complete MLOps lifecycle on the Titanic dataset: data validation and preprocessing, model training and comparison, experiment tracking and a model registry (MLflow), data and pipeline versioning (DVC), an automated CI/CD pipeline with a quality gate, a secured REST API (FastAPI), and drift and performance monitoring (Evidently AI).

> **About the scenario.** The assignment brief frames the work as hospital readmission risk, but the dataset it supplies is the Titanic passenger data. This project uses the supplied data as given: the binary target is `Survived`, and each "patient record" sent to the API is a passenger record. Nothing in the pipeline is specific to Titanic. Swapping in a readmission dataset means changing the column lists in `src/config.py`, the checks in `src/data.py`, and the features in `src/features.py`.

## Results at a glance

| Item | Value |
|---|---|
| Data | 891 rows, 12 columns, 38.4% positive class |
| Split | 703 train / 188 test (stable hash split on `PassengerId`) |
| Candidates | Logistic Regression, Random Forest, XGBoost (5-fold CV) |
| Selected model | XGBoost (best mean CV ROC-AUC, 0.882) |
| Test metrics | Accuracy 0.835, Precision 0.855, Recall 0.707, F1 0.774, ROC-AUC 0.849 |
| Tests | 32 automated tests (data, features, model, API) |

Full write-up: [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md).

## Project layout

```
titanic-mlops/
├── params.yaml               all settings: split, models, quality gate, monitoring thresholds
├── dvc.yaml / dvc.lock       DVC pipeline definition and the exact hashes of the last run
├── data/raw/titanic.csv      raw data (tracked by DVC through titanic.csv.dvc)
├── src/
│   ├── config.py             paths, column lists, parameter loading
│   ├── data.py               load, validate, clean, split
│   ├── features.py           feature engineering + preprocessing pipeline
│   ├── eda.py                exploratory analysis, writes figures and a JSON summary
│   ├── train.py              trains candidates, logs to MLflow, applies the gate, promotes
│   └── monitor.py            Evidently drift and performance reports, alerts
├── api/main.py               FastAPI service
├── pipeline/
│   ├── run_pipeline.py       the CI/CD pipeline (validate, test, train, gate, deploy, monitor)
│   └── trigger.py            re-runs the pipeline when data, code, or params change
├── tests/                    pytest suite
├── .github/workflows/ci.yml  GitHub Actions workflow
├── Dockerfile                container image for the API
├── models/                   serving artifact and its metadata
├── reports/                  metrics, EDA figures, Evidently HTML reports, pipeline run records
└── docs/                     project report, console evidence, screenshots
```

## 1. Setup

Requires Python 3.10 or newer and Git.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

git init                         # DVC and the pipeline both record the Git commit
dvc init                         # skip if .dvc/ already exists
dvc add data/raw/titanic.csv     # put the raw data under DVC version control
git add -A && git commit -m "Initial commit"
```

`make setup` does the same thing in one step.

## 2. Run the pipeline

```bash
python -m pipeline.run_pipeline
```

This runs six steps in order and stops at the first failure:

| Step | What it does | Fails when |
|---|---|---|
| 1. Validate and prepare | Schema and range checks on the raw file, cleaning, train/test split | A column is missing or values are out of range |
| 2. Unit tests | `tests/test_data.py`, `tests/test_features.py` | Any test fails |
| 3. EDA | Summary statistics and figures to `reports/eda/` | |
| 4. Train, gate, promote | Trains three candidates, logs each to MLflow, picks the best on CV, applies the quality gate | The model misses the gate |
| 5. Model and API tests | Tests the promoted artifact and exercises the API in-process | Any test fails |
| 6. Monitoring | Evidently drift and performance reports to `reports/monitoring/` | |

Each run writes a JSON record (Git commit, data hash, step results) to `reports/pipeline_runs/`.

### Individual stages

```bash
python -m src.data       # validate, clean, split
python -m src.eda        # exploratory analysis
python -m src.train      # train + MLflow + gate + promote
python -m src.monitor    # monitoring reports
python -m pytest -q      # all tests
```

## 3. Model and data versioning

**MLflow** tracks every training run and owns the model registry.

- Each candidate is one run with its hyperparameters, 5-fold CV metrics, and test metrics.
- Each run is tagged with the Git commit and the MD5 of the raw data, so any model traces back to exact code and data.
- The selected model is logged as an artifact (with a signature and input example) and registered as a new version of `titanic-survival-classifier`.
- The alias `champion` points to the version currently in production. A version that fails the gate is still registered (tagged `gate_passed=False`) so it can be inspected, but the alias does not move.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://localhost:5000
```

The default store is a local SQLite file. To share runs with a team, set `MLFLOW_TRACKING_URI` to a tracking server. No code changes are needed.

**DVC** versions the data and the pipeline.

- `data/raw/titanic.csv.dvc` holds the hash of the raw data. Git stores this small pointer file. The data itself goes to DVC storage.
- `dvc.yaml` declares each stage with its dependencies, parameters, and outputs. `dvc.lock` records the hash of every input and output from the last run.
- `dvc repro` re-runs only the stages whose inputs changed. `dvc metrics show` and `dvc params diff` compare runs across Git commits.

```bash
dvc repro                                  # reproduce the pipeline
dvc remote add -d storage /path/or/s3://bucket
dvc push                                   # upload data and models to the remote
git checkout <commit> && dvc checkout      # restore the data and model from any commit
```

## 4. CI/CD pipeline configuration

There are two layers, and both call the same `pipeline/run_pipeline.py`.

**Local trigger (`pipeline/trigger.py`).** Computes a fingerprint from the MD5 of the raw data, `params.yaml`, and every Python file in `src/`, `api/`, and `pipeline/`. If the fingerprint differs from the last successful run, the pipeline runs. The fingerprint is saved only after a success, so a failed run is retried next time.

```bash
python -m pipeline.trigger              # run only if something changed
python -m pipeline.trigger --force      # run regardless
python -m pipeline.trigger --watch 30   # poll every 30 seconds (or schedule with cron)
```

**GitHub Actions (`.github/workflows/ci.yml`).** Triggers on:

- a push or pull request that touches `data/`, `src/`, `api/`, `pipeline/`, `tests/`, `params.yaml`, or `requirements.txt`
- a weekly schedule (Mondays 06:00 UTC) for a routine retrain and drift check
- a manual run from the Actions tab

It installs dependencies, runs the pipeline, and uploads `models/` and `reports/` as build artifacts. Hosted runners start with an empty disk, so set `MLFLOW_TRACKING_URI` (as a repository secret) to a persistent tracking server if you want the registry and the champion comparison to carry over between CI runs. On `main`, a second job builds the Docker image for the API after the pipeline passes.

**Quality gate (`params.yaml`, section `gate`).** A candidate is promoted only if all three hold:

1. test ROC-AUC is at least `min_roc_auc` (0.80)
2. test F1 is at least `min_f1` (0.70)
3. test ROC-AUC is no more than `max_regression` (0.02) below the current champion, which is re-scored on the same test rows

**Deployment.** Promotion moves the `champion` alias in MLflow and writes `models/model.joblib`. The API checks that file's modification time on each request and reloads it when it changes, so a successful pipeline run deploys the new model without restarting the service. If any step fails, the serving artifact is left untouched.

## 5. Serve the model

```bash
export API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Interactive documentation: http://localhost:8000/docs

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | open | liveness and readiness probe |
| `POST /predict` | API key | score one record |
| `POST /predict/batch` | API key | score up to 500 records |
| `GET /model-info` | API key | version, Git commit, data hash, and test metrics of the serving model |

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"Pclass": 1, "Sex": "female", "Age": 38, "SibSp": 1, "Parch": 0, "Fare": 71.28, "Embarked": "C"}'
```

```json
{"survived": 1, "survival_probability": 0.9773, "risk_level": "low", "model_version": 3}
```

Only `Pclass` and `Sex` are required. Missing optional fields are imputed by the model pipeline. `risk_level` is the risk of the adverse outcome (low when survival probability is 0.65 or higher, high when below 0.35).

**Security measures**

- API key in the `X-API-Key` header, compared in constant time. The key comes from the `API_KEY` environment variable and is never stored in code.
- Fails closed: if `API_KEY` is not set, prediction endpoints return 503 instead of running unprotected.
- Strict input validation with Pydantic: allowed values, numeric ranges, string length limits, and unknown fields are rejected (422).
- Batch size capped at 500 records.
- Internal errors return a generic message. Stack traces are logged server-side only.
- The prediction log used for monitoring does not store the free-text `Name` field.
- The Docker image runs as a non-root user.
- For production, put the service behind HTTPS (a reverse proxy or API gateway) and add rate limiting there.

**Docker**

```bash
docker build -t titanic-api .
docker run -p 8000:8000 -e API_KEY="$API_KEY" titanic-api
```

## 6. Monitoring strategy

`python -m src.monitor` compares batches of current data against the training data (the reference) using Evidently AI.

| Batch | Source | Checks |
|---|---|---|
| `holdout` | untouched test split | data drift, prediction drift, classification performance |
| `simulated_drift` | test split with deliberate shifts (age +15, fare x2.5, half the rows moved to first class, 40% re-embarked at C) | same as above. This batch is synthetic and exists to prove the alert path works |
| `production` | requests logged by the API in `monitoring/prediction_log.jsonl` | data drift and prediction drift (no labels are available at serving time) |

**What is measured**

- **Data drift** per feature: Kolmogorov-Smirnov test for numeric features, chi-square or Z-test for categorical ones, at p < 0.05.
- **Prediction drift**: the same test on the distribution of predicted probabilities. This is the early warning when labels arrive late.
- **Performance**: accuracy, precision, recall, F1, and ROC-AUC whenever labels are available.

**Alerts (`params.yaml`, section `monitoring`)**

| Alert | Condition |
|---|---|
| `data_drift` | share of drifted features is at least `drift_share_threshold` (0.30) |
| `prediction_drift` | predicted-probability distribution shifted (p < 0.05) |
| `performance_drop` | F1 is more than `max_f1_drop` (0.05) below the F1 recorded at promotion |

**Outputs in `reports/monitoring/`**

- `<batch>_report.html`: the interactive Evidently report
- `monitoring_summary.json`: drift share, per-feature p-values, and metrics for each batch
- `alerts.json`: every alert raised, with severity and a plain-language message

`python -m src.monitor --fail-on-alert` exits with status 1 when a non-synthetic batch raises an alert. Use it in a scheduled job so that drift fails the job and notifies the team, or wire `alerts.json` to email or Slack.

**Response playbook.** Data drift without a performance drop: investigate the upstream source and watch closely. Prediction drift or a performance drop: add the newly labelled data to `data/raw/`, and the trigger retrains automatically. The quality gate decides whether the new model replaces the champion.

## Tools and configuration

| Tool | Role | Configuration |
|---|---|---|
| pandas, scikit-learn, XGBoost | data handling, preprocessing, models | `params.yaml` (`data`, `train`) |
| MLflow | experiment tracking, model registry | `MLFLOW_TRACKING_URI` (default `sqlite:///mlflow.db`) |
| DVC | data and pipeline versioning | `dvc.yaml`, `dvc.lock`, `.dvc/config` |
| GitPython | records commit, branch, and dirty state in each run | used in `pipeline/run_pipeline.py`, `src/train.py` |
| pytest | data, feature, model, and API tests | `tests/` |
| FastAPI, Uvicorn, Pydantic | REST API, validation | `API_KEY` environment variable |
| Evidently AI | drift and performance monitoring | `params.yaml` (`monitoring`) |
| GitHub Actions | hosted CI/CD | `.github/workflows/ci.yml` |
| Docker | packaging the API | `Dockerfile` |

## Best practices followed

**Reproducibility**

- One source of settings (`params.yaml`). No magic numbers in the code.
- Fixed random seeds for cross-validation and every model.
- Deterministic train/test split keyed on a hash of `PassengerId`, so a row stays in the same split on every run, including after new data is appended.
- All learned preprocessing (imputation values, outlier caps, scaling, category lists) lives inside one scikit-learn `Pipeline`. It is fitted on training data only and the identical object is used at serving time, which rules out train/serve skew and test-set leakage.
- Every model version is tagged with its Git commit and data hash.
- Model selection uses cross-validation on the training split. The test split is used once, for the final report and the gate.

**Scalability and operations**

- Stages are independent modules with file inputs and outputs, so they can move to separate workers or an orchestrator (Airflow, Prefect) without rewriting.
- DVC caching skips stages whose inputs did not change.
- The MLflow store and DVC remote are both switched by configuration, from local files to a shared server or object storage.
- The API is stateless and containerised, so it scales horizontally behind a load balancer.
- Batch endpoint for high-volume scoring.
- Failed candidates never replace the serving model, and the previous version stays in the registry for rollback: `MlflowClient().set_registered_model_alias(name, "champion", <old version>)`.

## Known limitations

- 891 rows is small. The three candidates are within one standard deviation of each other on cross-validated ROC-AUC, so the choice between them is not statistically meaningful.
- The production monitoring batch in `reports/` comes from demo traffic replayed from the dataset, since there is no real production traffic.
- An API key is the minimum viable access control. A real healthcare deployment would need per-client credentials, TLS, audit logging, and a privacy review.

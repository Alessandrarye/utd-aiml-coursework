# Incremental Capstone: MLOps for Bank Customer Churn Prediction

End-to-end MLOps solution for predicting customer churn at a bank: data preparation, model training, experiment tracking with MLflow, deployment with FastAPI and Docker, and simulated monitoring with automatic retraining.

## Project layout

```
incremental-capstone-mlops/
├── MLOps_Churn_Capstone.ipynb   # Walkthrough of all four tasks, with outputs
├── data/Churn_Modeling2.csv     # 10,000 bank customers
├── src/
│   ├── config.py                # Paths, feature lists, thresholds
│   ├── data_prep.py             # Task 1: load, clean, encode, scale
│   ├── train.py                 # Tasks 2 and 3: train, evaluate, log to MLflow, register
│   ├── monitor.py               # Task 4: simulated monitoring, drift checks, retraining
│   └── retrain.py               # Task 4: scheduled retraining check
├── app/main.py                  # Task 4: FastAPI prediction service
├── Dockerfile                   # Task 4: container for the API
├── requirements.txt
└── models/                      # Created by training: model, metadata, pinned serving requirements
```

## Quick start

```bash
pip install -r requirements.txt

# Option A: run the notebook top to bottom (does everything below)
# Option B: run the pieces from the command line
python -m src.train        # train 4 models, log to MLflow, register and save the best
python -m src.monitor      # simulate 8 weeks of production with drift and retraining
python -m src.retrain      # scheduled check: retrain only if monitoring says so
```

## MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open http://localhost:5000, select the `churn-prediction` experiment, tick the runs, and click **Compare**. Registered versions are under the **Models** tab as `churn-prediction-model`.

## API

```bash
uvicorn app.main:app --reload --port 8000
```

Docs at http://localhost:8000/docs.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"CreditScore": 619, "Geography": "France", "Gender": "Female", "Age": 42, "Tenure": 2,
       "Balance": 0, "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 101348.88}'
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `GET /model-info` | Model name, version, and metrics |
| `POST /predict` | Prediction for one customer |
| `POST /predict/batch` | Predictions for up to 1,000 customers |
| `GET /stats` | Predictions served and predicted churn rate |

## Docker

Train first so `models/` exists, then:

```bash
# Match PYTHON_VERSION to "python_version" in models/metadata.json (major.minor)
docker build --build-arg PYTHON_VERSION=3.12 -t churn-api .
docker run -d --rm --name churn-api -p 8000:8000 churn-api
curl http://localhost:8000/health
docker stop churn-api
```

## Monitoring and retraining

`src/monitor.py` cuts the held-out test set into 8 weekly batches and injects data drift and concept drift from week 4. Each week it records accuracy, F1, and PSI per feature (to `logs/monitoring_log.csv` and the `churn-monitoring` MLflow experiment). If accuracy stays more than 5 points below the baseline for 2 weeks in a row, the model is retrained and the new version is registered.

Cron schedule for a real deployment:

```cron
0 2 * * 1  cd /path/to/incremental-capstone-mlops && python -m src.monitor >> logs/monitor.log 2>&1
0 3 * * 1  cd /path/to/incremental-capstone-mlops && python -m src.retrain >> logs/retrain.log 2>&1
0 4 1 * *  cd /path/to/incremental-capstone-mlops && python -m src.retrain --force >> logs/retrain.log 2>&1
```

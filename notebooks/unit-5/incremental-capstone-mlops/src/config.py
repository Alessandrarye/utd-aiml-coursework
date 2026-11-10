"""Shared paths and settings for the churn MLOps project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Churn_Modeling2.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
SERVING_REQS_PATH = MODEL_DIR / "serving-requirements.txt"
LOG_DIR = ROOT / "logs"
MONITOR_LOG_PATH = LOG_DIR / "monitoring_log.csv"
PREDICTION_LOG_PATH = LOG_DIR / "predictions.jsonl"

# MLflow: a SQLite backend so the Model Registry works locally.
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
ARTIFACT_ROOT = (ROOT / "mlruns").as_uri()
EXPERIMENT_NAME = "churn-prediction"
MONITOR_EXPERIMENT_NAME = "churn-monitoring"
REGISTERED_MODEL_NAME = "churn-prediction-model"

TARGET = "Exited"
CATEGORICAL_FEATURES = ["Geography", "Gender"]
NUMERIC_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Monitoring: retrain when batch accuracy falls this far below the baseline.
ACCURACY_DROP_TOLERANCE = 0.05
# Require this many low weeks in a row, so one noisy batch does not trigger it.
RETRAIN_PATIENCE = 2
PSI_ALERT_THRESHOLD = 0.20

"""Paths, constants, and parameter loading shared by every stage."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
MODEL_META_PATH = MODEL_DIR / "model_meta.json"

REPORTS_DIR = ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"
FIGURES_DIR = REPORTS_DIR / "figures"
EDA_DIR = REPORTS_DIR / "eda"
MONITORING_DIR = REPORTS_DIR / "monitoring"
PIPELINE_RUNS_DIR = REPORTS_DIR / "pipeline_runs"

PREDICTION_LOG_PATH = ROOT / "monitoring" / "prediction_log.jsonl"

# MLflow uses a local SQLite backend by default so the model registry works
# without a server. Point MLFLOW_TRACKING_URI at a remote server to share runs.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")

TARGET = "Survived"

# Columns expected in the raw file.
RAW_COLUMNS = [
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
]

# Raw fields the model consumes. The API accepts exactly these.
INPUT_FEATURES = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]

# Columns produced by feature engineering, grouped by how they are transformed.
NUMERIC_FEATURES = ["Age", "FareLog", "SibSp", "Parch", "FamilySize", "Pclass"]
BINARY_FEATURES = ["IsAlone", "HasCabin", "AgeMissing"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title", "Deck"]


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)

"""Task 4 (steps 1 and 3): FastAPI service for real-time churn prediction.

Run locally from the project root:
    uvicorn app.main:app --reload --port 8000
Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "churn_model.joblib"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", ROOT / "models" / "metadata.json"))
PREDICTION_LOG = Path(os.getenv("PREDICTION_LOG", ROOT / "logs" / "predictions.jsonl"))

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup."""
    if MODEL_PATH.exists():
        state["model"] = joblib.load(MODEL_PATH)
        state["metadata"] = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    state["served"] = 0
    state["predicted_churn"] = 0
    yield
    state.clear()


app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description="Predicts whether a bank customer is likely to leave.",
    version="1.0.0",
    lifespan=lifespan,
)


class Customer(BaseModel):
    CreditScore: int = Field(..., ge=300, le=900, examples=[619])
    Geography: Literal["France", "Germany", "Spain"] = Field(..., examples=["France"])
    Gender: Literal["Male", "Female"] = Field(..., examples=["Female"])
    Age: int = Field(..., ge=18, le=100, examples=[42])
    Tenure: int = Field(..., ge=0, le=50, description="Years with the bank", examples=[2])
    Balance: float = Field(..., ge=0, examples=[0.0])
    NumOfProducts: int = Field(..., ge=1, le=10, examples=[1])
    HasCrCard: Literal[0, 1] = Field(..., examples=[1])
    IsActiveMember: Literal[0, 1] = Field(..., examples=[1])
    EstimatedSalary: float = Field(..., ge=0, examples=[101348.88])


class Prediction(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str
    model_name: str | None = None
    model_version: int | None = None


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(..., min_length=1, max_length=1000)


def _risk(p: float) -> str:
    return "high" if p >= 0.70 else "medium" if p >= 0.40 else "low"


def _predict(customers: list[Customer]) -> list[Prediction]:
    if "model" not in state:
        raise HTTPException(503, "Model not loaded. Run `python -m src.train` first.")
    meta = state["metadata"]
    X = pd.DataFrame([c.model_dump() for c in customers])[meta.get("features", list(Customer.model_fields))]
    probs = state["model"].predict_proba(X)[:, 1]
    preds = state["model"].predict(X)

    results = [
        Prediction(
            churn_prediction=int(pred), churn_probability=round(float(p), 4), risk_level=_risk(float(p)),
            model_name=meta.get("model_name"), model_version=meta.get("registered_version"),
        )
        for pred, p in zip(preds, probs)
    ]
    _log(customers, results)
    state["served"] += len(results)
    state["predicted_churn"] += int(sum(r.churn_prediction for r in results))
    return results


def _log(customers: list[Customer], results: list[Prediction]) -> None:
    """Append every request to a log file. This is the raw material for
    monitoring: it lets drift checks compare live inputs to training data."""
    try:
        PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with PREDICTION_LOG.open("a") as f:
            for c, r in zip(customers, results):
                f.write(json.dumps({"timestamp": ts, **c.model_dump(), **r.model_dump()}) + "\n")
    except OSError:
        pass  # never fail a prediction because logging failed


@app.get("/")
def root():
    return {"service": app.title, "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok" if "model" in state else "model_not_loaded"}


@app.get("/model-info")
def model_info():
    meta = {k: v for k, v in state.get("metadata", {}).items() if k != "reference_stats"}
    return meta or {"detail": "no model metadata found"}


@app.get("/stats")
def stats():
    """Simple live counters for monitoring the service."""
    served = state.get("served", 0)
    return {
        "predictions_served": served,
        "predicted_churn_rate": round(state.get("predicted_churn", 0) / served, 4) if served else None,
    }


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    return _predict([customer])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(request: BatchRequest):
    return _predict(request.customers)

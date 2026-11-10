"""FastAPI service that serves the promoted model.

Run locally:
    export API_KEY=choose-a-long-random-string
    uvicorn api.main:app --host 0.0.0.0 --port 8000
Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import threading
from datetime import datetime, timezone
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from src.config import INPUT_FEATURES, MODEL_META_PATH, MODEL_PATH, PREDICTION_LOG_PATH

logger = logging.getLogger("titanic-api")

app = FastAPI(
    title="Titanic Survival Risk API",
    version="1.0.0",
    description="Serves the champion classifier from the MLOps pipeline. "
                "All prediction endpoints require an X-API-Key header.",
)

# ---------------------------------------------------------------- security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: Optional[str] = Security(api_key_header)) -> None:
    expected = os.getenv("API_KEY")
    if not expected:
        # Fail closed: an unconfigured server never serves predictions.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "API key is not configured on the server")
    if not key or not secrets.compare_digest(key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")


# ------------------------------------------------------------------ schemas
class Passenger(BaseModel):
    """One record to score. Unknown fields are rejected, ranges are enforced."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1, "Parch": 0,
        "Fare": 7.25, "Embarked": "S", "Name": "Braund, Mr. Owen Harris", "Cabin": None}})

    Pclass: Literal[1, 2, 3]
    Sex: Literal["male", "female"]
    Age: Optional[float] = Field(None, ge=0, le=120)
    SibSp: int = Field(0, ge=0, le=20)
    Parch: int = Field(0, ge=0, le=20)
    Fare: Optional[float] = Field(None, ge=0, le=1000)
    Embarked: Optional[Literal["C", "Q", "S"]] = None
    Name: Optional[str] = Field(None, max_length=200)
    Cabin: Optional[str] = Field(None, max_length=30)


class Prediction(BaseModel):
    survived: int
    survival_probability: float
    risk_level: Literal["low", "medium", "high"]
    model_version: int


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[Passenger] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


# -------------------------------------------------------------- model store
class ModelStore:
    """Loads the serving artifact and reloads it when the pipeline promotes a new one."""

    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._meta: dict = {}
        self._mtime = None

    def get(self):
        if not MODEL_PATH.exists():
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "No model has been promoted yet")
        mtime = MODEL_PATH.stat().st_mtime
        if self._model is None or mtime != self._mtime:
            with self._lock:
                self._model = joblib.load(MODEL_PATH)
                self._meta = json.loads(MODEL_META_PATH.read_text()) if MODEL_META_PATH.exists() else {}
                self._mtime = mtime
                logger.info("Loaded model version %s", self._meta.get("model_version"))
        return self._model, self._meta


store = ModelStore()
_log_lock = threading.Lock()


def _risk(p_survive: float) -> str:
    """Risk of the adverse outcome (not surviving)."""
    if p_survive >= 0.65:
        return "low"
    return "medium" if p_survive >= 0.35 else "high"


def _log_predictions(frame: pd.DataFrame, probas, version) -> None:
    """Append scored requests for drift monitoring. Free-text Name is not stored."""
    try:
        PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = frame.drop(columns=["Name"]).assign(
            prediction=[int(p >= 0.5) for p in probas], probability=[round(float(p), 6) for p in probas],
            model_version=version, timestamp=ts)
        with _log_lock, open(PREDICTION_LOG_PATH, "a") as f:
            f.write(rows.to_json(orient="records", lines=True).rstrip("\n") + "\n")
    except Exception:
        logger.exception("Could not write prediction log")  # logging must never break serving


def _score(records: list[Passenger]) -> list[Prediction]:
    model, meta = store.get()
    frame = pd.DataFrame([r.model_dump() for r in records])[INPUT_FEATURES]
    try:
        probas = model.predict_proba(frame)[:, 1]
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Prediction failed")  # no internals leaked
    version = int(meta.get("model_version", 0))
    _log_predictions(frame, probas, version)
    return [Prediction(survived=int(p >= 0.5), survival_probability=round(float(p), 4),
                       risk_level=_risk(float(p)), model_version=version) for p in probas]


# ---------------------------------------------------------------- endpoints
@app.get("/health", tags=["ops"])
def health():
    """Liveness and readiness probe. Open, and reveals nothing sensitive."""
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/model-info", tags=["ops"], dependencies=[Depends(require_api_key)])
def model_info():
    _, meta = store.get()
    return meta


@app.post("/predict", response_model=Prediction, tags=["predict"], dependencies=[Depends(require_api_key)])
def predict(passenger: Passenger):
    return _score([passenger])[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["predict"],
          dependencies=[Depends(require_api_key)])
def predict_batch(batch: BatchRequest):
    return BatchResponse(predictions=_score(batch.records))

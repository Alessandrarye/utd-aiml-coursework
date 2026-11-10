"""Tasks 2 and 3: train several churn models, track them in MLflow,
register the best one (by F1), and save the artifacts the API serves.

Run from the project root:
    python -m src.train
"""
from __future__ import annotations

import json
import logging
import platform
import warnings
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src import config
from src.data_prep import build_preprocessor, load_clean, split


def get_models() -> dict:
    """Candidate models. Churners are only ~20% of customers, so every model
    is trained with balanced class weights (see ``fit_pipeline``)."""
    rs = config.RANDOM_STATE
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, C=1.0, random_state=rs),
        "decision_tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=rs),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=rs
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=3, random_state=rs
        ),
    }


def fit_pipeline(model, X, y, sample_weight=None) -> Pipeline:
    """Preprocessing + model in one pipeline, so the API can take raw rows."""
    weights = compute_sample_weight("balanced", y)
    if sample_weight is not None:
        weights = weights * np.asarray(sample_weight)
    pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
    pipe.fit(X, y, model__sample_weight=weights)
    return pipe


def evaluate(pipe: Pipeline, X, y) -> dict:
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }


def setup_mlflow(experiment_name: str = config.EXPERIMENT_NAME) -> None:
    # Quiet MLflow's routine notices (integer-schema hints, pickle reminders).
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module="mlflow")
    mlflow.set_tracking_uri(config.TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=config.ARTIFACT_ROOT)
    mlflow.set_experiment(experiment_name)


def _log_model(pipe, X_example):
    """Log a model to MLflow. Works on both MLflow 2.x and 3.x."""
    signature = infer_signature(X_example, pipe.predict(X_example))
    # cloudpickle keeps this working on every MLflow version (newer releases
    # default to skops, which refuses tree models unless types are whitelisted).
    kwargs = dict(signature=signature, input_example=X_example.head(3),
                  serialization_format="cloudpickle")
    try:
        return mlflow.sklearn.log_model(pipe, name="model", **kwargs)
    except TypeError:  # MLflow 2.x uses artifact_path
        return mlflow.sklearn.log_model(pipe, artifact_path="model", **kwargs)


def reference_stats(X: pd.DataFrame) -> dict:
    """Training-set distributions, saved so monitoring can measure drift."""
    stats = {"numeric": {}, "categorical": {}}
    for col in config.NUMERIC_FEATURES:
        edges = np.unique(np.quantile(X[col], np.linspace(0, 1, 11)))
        stats["numeric"][col] = {
            "mean": float(X[col].mean()),
            "std": float(X[col].std()),
            "bin_edges": [float(e) for e in edges],
        }
    for col in config.CATEGORICAL_FEATURES:
        stats["categorical"][col] = X[col].value_counts(normalize=True).round(6).to_dict()
    return stats


def train_and_track(
    X_train, y_train, X_test, y_test,
    sample_weight=None, run_tag: str = "initial_training", register: bool = True, deploy: bool = True,
) -> dict:
    """Train every candidate, log each as its own MLflow run, register and
    save the best one by F1. Returns a summary dict.

    ``deploy=False`` registers the new version in MLflow but leaves the model
    file the API serves, and the "champion" alias, untouched. The monitoring
    simulation uses that so a made-up drift scenario never replaces the real model."""
    setup_mlflow()
    results, pipes, infos = [], {}, {}

    for name, model in get_models().items():
        with mlflow.start_run(run_name=name) as run:
            pipe = fit_pipeline(model, X_train, y_train, sample_weight)
            metrics = evaluate(pipe, X_test, y_test)

            mlflow.set_tags({"stage": run_tag, "model_family": type(model).__name__})
            mlflow.log_params({k: v for k, v in model.get_params().items() if v is not None})
            mlflow.log_params({
                "class_weighting": "balanced",
                "n_train_rows": len(X_train),
                "n_test_rows": len(X_test),
                "features": ",".join(config.FEATURES),
            })
            mlflow.log_metrics(metrics)
            infos[name] = _log_model(pipe, X_test)

            pipes[name] = pipe
            results.append({"model": name, "run_id": run.info.run_id, **metrics})

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)
    best = results_df.iloc[0]
    best_name, best_pipe = best["model"], pipes[best["model"]]

    version = None
    if register:
        mv = mlflow.register_model(infos[best_name].model_uri, config.REGISTERED_MODEL_NAME)
        version = int(mv.version)
        client = MlflowClient()
        if deploy:
            client.set_registered_model_alias(config.REGISTERED_MODEL_NAME, "champion", mv.version)
        client.set_model_version_tag(config.REGISTERED_MODEL_NAME, mv.version, "stage", run_tag)
        client.set_model_version_tag(
            config.REGISTERED_MODEL_NAME, mv.version, "f1", f"{best['f1']:.4f}"
        )

    if deploy:
        save_artifacts(best_pipe, best_name, best, X_train, version, run_tag)
    return {
        "results": results_df, "best_name": best_name, "best_pipeline": best_pipe,
        "best_metrics": {k: float(best[k]) for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
        "registered_version": version,
    }


def save_artifacts(pipe, name, metrics_row, X_train, version, run_tag) -> None:
    """Save what the FastAPI service needs: the model, its metadata, and the
    exact library versions it was trained with (used by the Dockerfile)."""
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, config.MODEL_PATH, compress=3)

    metadata = {
        "model_name": name,
        "registered_model": config.REGISTERED_MODEL_NAME,
        "registered_version": version,
        "stage": run_tag,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": {k: round(float(metrics_row[k]), 4)
                    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
        "features": config.FEATURES,
        "n_train_rows": int(len(X_train)),
        "python_version": platform.python_version(),
        "reference_stats": reference_stats(X_train),
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    config.SERVING_REQS_PATH.write_text(
        "# Written by src/train.py. Pins the libraries the saved model needs.\n"
        f"scikit-learn=={sklearn.__version__}\n"
        f"pandas=={pd.__version__}\n"
        f"numpy=={np.__version__}\n"
        f"joblib=={joblib.__version__}\n"
        "fastapi\nuvicorn[standard]\n"
    )


def main() -> None:
    df = load_clean()
    X_train, X_test, y_train, y_test = split(df)
    out = train_and_track(X_train, y_train, X_test, y_test)
    print(out["results"].drop(columns="run_id").round(4).to_string(index=False))
    print(f"\nBest model by F1: {out['best_name']} "
          f"(registered as {config.REGISTERED_MODEL_NAME} v{out['registered_version']})")
    print(f"Saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()

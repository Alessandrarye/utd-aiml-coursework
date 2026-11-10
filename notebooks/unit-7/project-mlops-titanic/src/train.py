"""Train candidate models, track everything in MLflow, gate, and promote the winner."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (FIGURES_DIR, INPUT_FEATURES, METRICS_PATH, MLFLOW_TRACKING_URI, MODEL_DIR,
                        MODEL_META_PATH, MODEL_PATH, ROOT, TARGET, load_params)
from src.data import file_md5
from src.features import build_preprocessor

CHAMPION = "champion"


def git_sha() -> str:
    try:
        import git
        return git.Repo(ROOT, search_parent_directories=True).head.commit.hexsha
    except Exception:
        return "no-git"


def build_estimator(name: str, hp: dict, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(**hp, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(**hp, random_state=seed, n_jobs=-1)
    if name == "xgboost":
        return XGBClassifier(**hp, random_state=seed, eval_metric="logloss", n_jobs=-1)
    raise ValueError(f"unknown model: {name}")


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }


def save_figures(model, X_test, y_test, name: str) -> list:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues", colorbar=False,
                                          display_labels=["Did not survive", "Survived"])
    ax.set_title(f"Confusion matrix, test set ({name})", fontsize=10)
    fig.tight_layout(); p = FIGURES_DIR / "confusion_matrix.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], ls="--", color="#8a8f98", lw=1)
    ax.set_title("ROC curve, test set", fontsize=10)
    fig.tight_layout(); p = FIGURES_DIR / "roc_curve.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)
    return paths


def champion_metrics(client: MlflowClient, model_name: str, X_test, y_test):
    """Score the model that is currently serving on the current test split.

    The split is keyed on PassengerId (see src.data.split), so the champion has
    never trained on any of these rows and the comparison with the candidate is
    like for like. Returns None on the first run or if the champion cannot load.
    """
    try:
        mv = client.get_model_version_by_alias(model_name, CHAMPION)
        champion = mlflow.sklearn.load_model(f"models:/{model_name}@{CHAMPION}")
        return {"version": int(mv.version), "roc_auc": round(float(evaluate(champion, X_test, y_test)["roc_auc"]), 4)}
    except Exception:
        return None


def main() -> int:
    params = load_params()
    tp, gate, dp = params["train"], params["gate"], params["data"]
    seed = dp["random_state"]

    processed = ROOT / dp["processed_dir"]
    train = pd.read_csv(processed / "train.csv")
    test = pd.read_csv(processed / "test.csv")
    X_train, y_train = train[INPUT_FEATURES], train[TARGET]
    X_test, y_test = test[INPUT_FEATURES], test[TARGET]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(tp["experiment_name"])
    client = MlflowClient()

    data_md5 = file_md5(ROOT / dp["raw_path"])
    sha = git_sha()
    cv = StratifiedKFold(n_splits=tp["cv_folds"], shuffle=True, random_state=seed)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = {}

    # One MLflow run per candidate: parameters, CV metrics, and test metrics.
    for name, hp in tp["models"].items():
        model = Pipeline([("prep", build_preprocessor()), ("clf", build_estimator(name, hp, seed))])
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({"git_sha": sha, "data_md5": data_md5, "stage": "candidate"})
            mlflow.log_params({"model_type": name, **hp, "cv_folds": tp["cv_folds"],
                               "test_size": dp["test_size"], "random_state": seed,
                               "n_train": len(train), "n_test": len(test)})
            scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
            cv_metrics = {f"cv_{m}": float(scores[f"test_{m}"].mean()) for m in scoring}
            cv_metrics[f"cv_{tp['selection_metric']}_std"] = float(scores[f"test_{tp['selection_metric']}"].std())
            model.fit(X_train, y_train)
            test_metrics = {f"test_{k}": float(v) for k, v in evaluate(model, X_test, y_test).items()}
            mlflow.log_metrics({**cv_metrics, **test_metrics})
            results[name] = {"run_id": run.info.run_id, "model": model, **cv_metrics, **test_metrics}
            print(f"{name:20s} cv_roc_auc={cv_metrics['cv_roc_auc']:.4f} cv_f1={cv_metrics['cv_f1']:.4f} "
                  f"test_roc_auc={test_metrics['test_roc_auc']:.4f} test_f1={test_metrics['test_f1']:.4f}")

    # Select on cross-validation, never on the test set, so the test score stays honest.
    best_name = max(results, key=lambda n: results[n][f"cv_{tp['selection_metric']}"])
    best = results[best_name]
    model = best["model"]

    previous = champion_metrics(client, tp["registered_model_name"], X_test, y_test)
    checks = {
        "roc_auc_above_min": best["test_roc_auc"] >= gate["min_roc_auc"],
        "f1_above_min": best["test_f1"] >= gate["min_f1"],
        "no_regression_vs_champion": previous is None
        or best["test_roc_auc"] >= previous["roc_auc"] - gate["max_regression"],
    }
    passed = all(checks.values())

    # Log the winning model artifact and register it as a new version either way,
    # so a failed candidate is still inspectable. Only a pass moves the alias.
    with mlflow.start_run(run_id=best["run_id"]):
        for p in save_figures(model, X_test, y_test, best_name):
            mlflow.log_artifact(str(p), artifact_path="figures")
        mlflow.log_artifact(str(ROOT / "params.yaml"))
        info = mlflow.sklearn.log_model(
            model, name="model",
            signature=infer_signature(X_train, model.predict_proba(X_train)[:, 1]),
            input_example=X_train.head(3),
            code_paths=[str(ROOT / "src")],
            # The pipeline contains our own transformers (src.features), so use
            # cloudpickle. Only load model artifacts from a registry you trust.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            registered_model_name=tp["registered_model_name"],
        )
        mlflow.set_tags({"stage": "selected", "gate_passed": str(passed)})
    version = info.registered_model_version
    client.set_model_version_tag(tp["registered_model_name"], version, "gate_passed", str(passed))
    client.set_model_version_tag(tp["registered_model_name"], version, "git_sha", sha)
    client.set_model_version_tag(tp["registered_model_name"], version, "data_md5", data_md5)

    summary = {
        "selected_model": best_name,
        "model_version": int(version),
        "gate_passed": passed,
        "gate_checks": checks,
        "previous_champion": previous,
        "test": {k.replace("test_", ""): round(v, 4) for k, v in best.items() if k.startswith("test_")},
        "cv": {k.replace("cv_", ""): round(v, 4) for k, v in best.items() if k.startswith("cv_")},
        "candidates": {n: {k: round(v, 4) for k, v in r.items() if k.startswith(("cv_", "test_"))}
                       for n, r in results.items()},
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(summary, indent=2))

    if not passed:
        failed = [k for k, ok in checks.items() if not ok]
        print(f"QUALITY GATE FAILED on {failed}. Test metrics: {summary['test']}. "
              f"Version {version} registered but NOT promoted.")
        return 1

    # Promote: move the registry alias and export the serving artifact.
    client.set_registered_model_alias(tp["registered_model_name"], CHAMPION, version)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    MODEL_META_PATH.write_text(json.dumps({
        "model_name": tp["registered_model_name"],
        "model_version": int(version),
        "model_type": best_name,
        "mlflow_run_id": best["run_id"],
        "git_sha": sha,
        "data_md5": data_md5,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "test_metrics": summary["test"],
        "input_features": INPUT_FEATURES,
    }, indent=2))
    print(f"Promoted {best_name} as version {version} (alias '{CHAMPION}'). Test metrics: {summary['test']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

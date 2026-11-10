"""Task 4 (steps 4 and 5): simulated monitoring, drift detection, and
automatic retraining when accuracy drops.

There is no live bank traffic for a class project, so production is simulated:
the held-out test set is cut into weekly batches, and from a chosen week on the
customers start behaving differently (data drift plus concept drift). Each week
the deployed model is scored on the new batch. If accuracy falls more than
ACCURACY_DROP_TOLERANCE below the baseline for RETRAIN_PATIENCE weeks in a
row, the model is retrained on the
original data plus the labeled batches from those weeks, and the new version is
registered in MLflow.

Run from the project root:
    python -m src.monitor
"""
from __future__ import annotations

import json

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src import config
from src.data_prep import load_clean, split
from src.train import setup_mlflow, train_and_track


# ---------------------------------------------------------------- drift math
def psi(expected: np.ndarray, actual: np.ndarray, bin_edges: list[float]) -> float:
    """Population Stability Index between training data and a new batch.
    Rule of thumb: < 0.10 stable, 0.10 to 0.20 moderate shift, > 0.20 major shift."""
    edges = np.array(bin_edges, dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(X_ref: pd.DataFrame, X_new: pd.DataFrame, ref_stats: dict) -> dict:
    """PSI for each numeric feature that has enough distinct values to bin."""
    report = {}
    for col, st in ref_stats["numeric"].items():
        if len(st["bin_edges"]) >= 4:
            report[col] = round(psi(X_ref[col].to_numpy(), X_new[col].to_numpy(), st["bin_edges"]), 4)
    return report


# ------------------------------------------------------- simulated production
def simulate_batches(
    X_test: pd.DataFrame, y_test: pd.Series,
    n_batches: int = 8, drift_start: int = 4, seed: int = config.RANDOM_STATE,
):
    """Cut the test set into ``n_batches`` disjoint weekly batches.

    Weeks before ``drift_start`` are untouched. From ``drift_start`` on, a
    scenario plays out: a competitor targets inactive customers under 45, so
    that group starts leaving at a much higher rate (concept drift), while the
    incoming customer mix skews older with higher balances (data drift).
    The effect is strongest from its first week, like a real market shock.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X_test))
    for week, idx in enumerate(np.array_split(order, n_batches), start=1):
        X = X_test.iloc[idx].copy().reset_index(drop=True)
        y = y_test.iloc[idx].copy().reset_index(drop=True)
        drifted = week >= drift_start
        if drifted:
            # Concept drift: the relationship between features and churn changes.
            target_group = (X["IsActiveMember"] == 0) & (X["Age"] < 45)
            leaves = target_group & (rng.random(len(X)) < 0.9)
            y[leaves] = 1
            loyal_now = (X["Age"] >= 50) & (X["IsActiveMember"] == 1) & (rng.random(len(X)) < 0.9)
            y[loyal_now] = 0
            # Data drift: the input distributions move.
            X["Age"] = (X["Age"] + rng.integers(2, 7, len(X))).clip(upper=92)
            X["Balance"] = np.where(X["Balance"] > 0, X["Balance"] * 1.15, X["Balance"]).round(2)
        yield week, X, y, drifted


def run_monitoring(n_batches: int = 8, drift_start: int = 4, log_to_mlflow: bool = True) -> pd.DataFrame:
    """Score each weekly batch, check drift, and retrain when accuracy drops."""
    df = load_clean()
    X_train, X_test, y_train, y_test = split(df)

    model = joblib.load(config.MODEL_PATH)
    meta = json.loads(config.METADATA_PATH.read_text())
    baseline_acc = meta["metrics"]["accuracy"]
    threshold = baseline_acc - config.ACCURACY_DROP_TOLERANCE
    model_version = meta.get("registered_version") or 1

    degraded_X, degraded_y, rows = [], [], []
    for week, Xb, yb, drifted in simulate_batches(X_test, y_test, n_batches, drift_start):
        pred = model.predict(Xb)
        acc, f1 = accuracy_score(yb, pred), f1_score(yb, pred, zero_division=0)
        psi_scores = drift_report(X_train, Xb, meta["reference_stats"])
        max_psi = max(psi_scores.values())

        # Labels arrive after the fact. Batches where the model underperformed
        # are kept as evidence of the new behavior and used for retraining.
        below = acc < threshold
        if below:
            degraded_X.append(Xb)
            degraded_y.append(yb)
        else:
            degraded_X, degraded_y = [], []
        needs_retrain = len(degraded_X) >= config.RETRAIN_PATIENCE

        rows.append({
            "week": week, "n_rows": len(Xb), "model_version": model_version,
            "accuracy": round(acc, 4), "f1": round(f1, 4),
            "actual_churn_rate": round(float(yb.mean()), 4),
            "predicted_churn_rate": round(float(pred.mean()), 4),
            "max_psi": round(max_psi, 4),
            "most_drifted_feature": max(psi_scores, key=psi_scores.get),
            "drift_alert": max_psi > config.PSI_ALERT_THRESHOLD,
            "accuracy_threshold": round(threshold, 4),
            "below_threshold": below,
            "retrain_triggered": needs_retrain,
            "drift_injected": drifted,
        })

        if log_to_mlflow:
            setup_mlflow(config.MONITOR_EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"week_{week:02d}"):
                mlflow.set_tags({"model_version": model_version, "retrain_triggered": needs_retrain})
                mlflow.log_metrics({"accuracy": acc, "f1": f1, "max_psi": max_psi,
                                    **{f"psi_{k}": v for k, v in psi_scores.items()}})

        if needs_retrain:
            model, model_version = retrain(X_train, y_train, degraded_X, degraded_y, week)
            degraded_X, degraded_y = [], []

    log = pd.DataFrame(rows)
    config.LOG_DIR.mkdir(exist_ok=True)
    log.to_csv(config.MONITOR_LOG_PATH, index=False)
    return log


def retrain(X_train, y_train, degraded_X, degraded_y, week, recent_weight: float = 8.0):
    """Retrain on the original data plus the labeled batches where the model
    underperformed.

    The recent batches get a larger weight so the new model leans toward
    current customer behavior. A quarter of the recent rows are held out to
    validate the candidates, and the best one by F1 is registered in MLflow as
    a new version. Because this is a simulation, the new model is swapped in
    for the rest of the simulated weeks only (deploy=False), and the real
    model file that the API serves is left alone.
    """
    recent_X = pd.concat(degraded_X, ignore_index=True)
    recent_y = pd.concat(degraded_y, ignore_index=True)
    rX_train, rX_val, ry_train, ry_val = train_test_split(
        recent_X, recent_y, test_size=0.25, stratify=recent_y, random_state=config.RANDOM_STATE
    )
    X_new = pd.concat([X_train, rX_train], ignore_index=True)
    y_new = pd.concat([y_train, ry_train], ignore_index=True)
    weights = np.r_[np.ones(len(X_train)), np.full(len(rX_train), recent_weight)]

    out = train_and_track(
        X_new, y_new, rX_val, ry_val,
        sample_weight=weights, run_tag=f"simulated_retrain_week_{week}", deploy=False,
    )
    return out["best_pipeline"], out["registered_version"]


def main() -> None:
    log = run_monitoring()
    cols = ["week", "model_version", "accuracy", "f1", "max_psi", "drift_alert",
            "below_threshold", "retrain_triggered"]
    print(log[cols].to_string(index=False))
    print(f"\nLog written to {config.MONITOR_LOG_PATH}")


if __name__ == "__main__":
    main()

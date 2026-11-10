"""Model monitoring with Evidently: data drift, prediction drift, and performance.

Reference data is the training split (what the model learned from). Each
"current" batch is compared with it:

  holdout          the untouched test split (labels available)
  simulated_drift  the test split with deliberate shifts applied, so the alert
                   path can be demonstrated and tested (clearly synthetic)
  production       requests logged by the API, once enough have accumulated
                   (no labels, so drift only)

Outputs go to reports/monitoring/: one interactive HTML report per batch,
monitoring_summary.json, and alerts.json.

    python -m src.monitor                  # write reports, always exit 0
    python -m src.monitor --fail-on-alert  # exit 1 if a real batch raised an alert
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import ClassificationPreset, DataDriftPreset

from src.config import INPUT_FEATURES, MODEL_META_PATH, MODEL_PATH, MONITORING_DIR, PREDICTION_LOG_PATH, ROOT, TARGET, load_params

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
log = logging.getLogger("monitor")

NUMERIC = ["Age", "Fare", "SibSp", "Parch"]
CATEGORICAL = ["Pclass", "Sex", "Embarked"]
MONITORED = NUMERIC + CATEGORICAL


def with_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    df = df.copy()
    df["probability"] = model.predict_proba(df[INPUT_FEATURES])[:, 1]
    df["prediction"] = (df["probability"] >= 0.5).astype(int)
    return df


def simulate_drift(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Synthetic shifted batch: an older, wealthier, first-class-heavy population."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["Age"] = out["Age"] + 15
    out["Fare"] = out["Fare"] * 2.5
    upgrade = rng.random(len(out)) < 0.5
    out.loc[upgrade, "Pclass"] = 1
    out.loc[rng.random(len(out)) < 0.4, "Embarked"] = "C"
    return out


def load_production_log(min_rows: int):
    if not PREDICTION_LOG_PATH.exists():
        return None
    df = pd.read_json(PREDICTION_LOG_PATH, lines=True)
    if len(df) < min_rows:
        log.info("Production log has %d rows, below the %d needed. Skipping.", len(df), min_rows)
        return None
    return df


def run_report(name: str, reference: pd.DataFrame, current: pd.DataFrame, has_labels: bool) -> dict:
    cols = MONITORED + ["prediction", "probability"] + ([TARGET] if has_labels else [])
    if has_labels:
        definition = DataDefinition(
            numerical_columns=NUMERIC + ["probability"], categorical_columns=CATEGORICAL,
            classification=[BinaryClassification(target=TARGET, prediction_labels="prediction",
                                                 prediction_probas="probability")])
        metrics = [DataDriftPreset(columns=MONITORED), ValueDrift(column="probability"), ClassificationPreset()]
    else:
        definition = DataDefinition(numerical_columns=NUMERIC + ["probability"], categorical_columns=CATEGORICAL)
        metrics = [DataDriftPreset(columns=MONITORED), ValueDrift(column="probability")]
        cols = MONITORED + ["probability"]

    ref = Dataset.from_pandas(reference[cols], data_definition=definition)
    cur = Dataset.from_pandas(current[cols], data_definition=definition)
    snapshot = Report(metrics).run(cur, ref)
    html_path = MONITORING_DIR / f"{name}_report.html"
    snapshot.save_html(str(html_path))

    result = {"batch": name, "rows": int(len(current)), "has_labels": has_labels,
              "report": str(html_path.relative_to(ROOT)), "drifted_columns": [], "column_p_values": {}}
    for m in snapshot.dict()["metrics"]:
        label, value = m.get("metric_name") or m.get("metric_id") or "", m["value"]
        if label.startswith("DriftedColumnsCount"):
            result["drift_share"] = round(float(value["share"]), 4)
        elif label.startswith("ValueDrift"):
            col = re.search(r"column=([^,)]+)", label).group(1)
            threshold = float(re.search(r"threshold=([0-9.]+)", label).group(1))
            if col == "probability":
                result["prediction_drift_p_value"] = round(float(value), 6)
                result["prediction_drift"] = bool(value < threshold)
            else:
                result["column_p_values"][col] = round(float(value), 6)
                if value < threshold:
                    result["drifted_columns"].append(col)
        elif has_labels:
            names = {"Accuracy": "accuracy", "Precision": "precision", "Recall": "recall",
                     "F1Score": "f1", "RocAuc": "roc_auc"}
            for key, out_name in names.items():
                if label.startswith(key + "("):
                    result.setdefault("performance", {})[out_name] = round(float(value), 4)
    return result


def check_alerts(result: dict, baseline_f1: float, cfg: dict) -> list:
    alerts = []
    if result.get("drift_share", 0) >= cfg["drift_share_threshold"]:
        alerts.append({"batch": result["batch"], "type": "data_drift", "severity": "high",
                       "message": f"{result['drift_share']:.0%} of monitored features drifted "
                                  f"(threshold {cfg['drift_share_threshold']:.0%}): {', '.join(result['drifted_columns'])}"})
    if result.get("prediction_drift"):
        alerts.append({"batch": result["batch"], "type": "prediction_drift", "severity": "medium",
                       "message": f"Distribution of predicted probabilities shifted (p={result['prediction_drift_p_value']})"})
    f1 = result.get("performance", {}).get("f1")
    if f1 is not None and baseline_f1 - f1 > cfg["max_f1_drop"]:
        alerts.append({"batch": result["batch"], "type": "performance_drop", "severity": "high",
                       "message": f"F1 fell from {baseline_f1:.3f} at promotion to {f1:.3f} "
                                  f"(allowed drop {cfg['max_f1_drop']})"})
    return alerts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-alert", action="store_true",
                        help="exit 1 when a non-synthetic batch raises an alert")
    args = parser.parse_args()

    params = load_params()
    cfg = params["monitoring"]
    processed = ROOT / params["data"]["processed_dir"]
    model = joblib.load(MODEL_PATH)
    meta = json.loads(MODEL_META_PATH.read_text())
    baseline_f1 = meta["test_metrics"]["f1"]
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    reference = with_predictions(pd.read_csv(processed / "train.csv"), model)
    test = pd.read_csv(processed / "test.csv")

    batches = [
        ("holdout", with_predictions(test, model), True, False),
        ("simulated_drift", with_predictions(simulate_drift(test), model), True, True),
    ]
    production = load_production_log(cfg["min_production_rows"])
    if production is not None:
        batches.append(("production", production, False, False))

    results, alerts = [], []
    for name, current, has_labels, synthetic in batches:
        result = run_report(name, reference, current, has_labels)
        result["synthetic"] = synthetic
        batch_alerts = check_alerts(result, baseline_f1, cfg)
        for a in batch_alerts:
            a["synthetic"] = synthetic
            log.warning("ALERT [%s] %s: %s", a["batch"], a["type"], a["message"])
        result["alerts"] = len(batch_alerts)
        results.append(result)
        alerts.extend(batch_alerts)
        log.info("%s: drift_share=%s drifted=%s performance=%s", name, result.get("drift_share"),
                 result["drifted_columns"], result.get("performance"))

    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {"generated_at": generated, "model_version": meta["model_version"],
               "baseline_f1": baseline_f1, "thresholds": cfg, "batches": results}
    (MONITORING_DIR / "monitoring_summary.json").write_text(json.dumps(summary, indent=2))
    (MONITORING_DIR / "alerts.json").write_text(json.dumps({"generated_at": generated, "alerts": alerts}, indent=2))

    real_alerts = [a for a in alerts if not a["synthetic"]]
    log.info("Monitoring complete: %d alert(s), %d on real data.", len(alerts), len(real_alerts))
    return 1 if (args.fail_on_alert and real_alerts) else 0


if __name__ == "__main__":
    sys.exit(main())

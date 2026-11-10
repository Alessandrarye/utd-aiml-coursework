"""Task 4 (step 5): scheduled retraining check.

Meant to be run on a schedule (cron example in the README). It reads the
monitoring log and retrains only when accuracy has stayed below the threshold
for RETRAIN_PATIENCE checks in a row, or when --force is passed.

    python -m src.retrain            # retrain only if monitoring says so
    python -m src.retrain --force    # retrain no matter what

Any labeled CSVs dropped into data/new_batches/ (same columns as the original
file) are added to the training data.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src import config
from src.data_prep import clean, load_clean, split
from src.train import train_and_track


def retrain_needed() -> tuple[bool, str]:
    if not config.MONITOR_LOG_PATH.exists():
        return False, "no monitoring log yet"
    log = pd.read_csv(config.MONITOR_LOG_PATH)
    current = log[log["model_version"] == log["model_version"].iloc[-1]]
    recent = current.tail(config.RETRAIN_PATIENCE)
    if len(recent) < config.RETRAIN_PATIENCE:
        return False, "not enough checks on the current model version yet"
    low = (recent["accuracy"] < recent["accuracy_threshold"]).all()
    detail = f"last {config.RETRAIN_PATIENCE} accuracies: {recent['accuracy'].tolist()}, " \
             f"threshold {recent['accuracy_threshold'].iloc[-1]}"
    return bool(low), detail


def load_training_data() -> pd.DataFrame:
    frames = [load_clean()]
    for path in sorted((config.ROOT / "data" / "new_batches").glob("*.csv")):
        frames.append(clean(pd.read_csv(path)))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="retrain even if accuracy is fine")
    args = parser.parse_args()

    needed, detail = retrain_needed()
    print(f"Monitoring check: {detail}")
    if not (needed or args.force):
        print("No retraining needed.")
        return

    X_train, X_test, y_train, y_test = split(load_training_data())
    out = train_and_track(X_train, y_train, X_test, y_test, run_tag="scheduled_retrain")
    print(f"Retrained. Best model: {out['best_name']} "
          f"(F1 {out['best_metrics']['f1']:.4f}), registered as "
          f"{config.REGISTERED_MODEL_NAME} v{out['registered_version']}")


if __name__ == "__main__":
    main()

"""Loading, validation, cleaning, and splitting of the raw Titanic data."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from src.config import INPUT_FEATURES, RAW_COLUMNS, ROOT, TARGET, load_params


def file_md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else ROOT / load_params()["data"]["raw_path"]
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Return a list of data problems. An empty list means the data is usable.

    This is the data test that runs at the start of the pipeline, so a bad
    file stops the run before any model is trained on it.
    """
    problems: list[str] = []
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if len(df) < 100:
        problems.append(f"too few rows: {len(df)}")
    if df[TARGET].isna().any():
        problems.append("target has missing values")
    if not set(df[TARGET].dropna().unique()) <= {0, 1}:
        problems.append("target must be binary 0/1")
    if not set(df["Pclass"].dropna().unique()) <= {1, 2, 3}:
        problems.append("Pclass outside {1, 2, 3}")
    if not set(df["Sex"].dropna().unique()) <= {"male", "female"}:
        problems.append("Sex outside {male, female}")
    if (df["Age"].dropna() < 0).any() or (df["Age"].dropna() > 120).any():
        problems.append("Age outside 0 to 120")
    if (df["Fare"].dropna() < 0).any():
        problems.append("negative Fare")
    if df["Age"].isna().mean() > 0.5:
        problems.append("more than half of Age is missing")
    return problems


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning that is safe to do before the train/test split.

    Anything that learns from the data (imputation values, outlier caps,
    category lists) lives in the model pipeline instead, so it is fitted on
    the training split only and applied identically at serving time.
    """
    df = df.copy()
    df = df.drop_duplicates(subset="PassengerId")
    df = df.dropna(subset=[TARGET])
    for col in ["Name", "Sex", "Cabin", "Embarked"]:
        df[col] = df[col].astype("string").str.strip()
    df["Sex"] = df["Sex"].str.lower()
    df["Embarked"] = df["Embarked"].str.upper()
    df[TARGET] = df[TARGET].astype(int)
    # Ticket is a near-unique string that does not generalise. PassengerId is kept
    # only as a row key for splitting. It is never used as a model feature.
    return df[["PassengerId"] + INPUT_FEATURES + [TARGET]].reset_index(drop=True)


def split(df: pd.DataFrame, test_size: float):
    """Deterministic split on a hash of PassengerId.

    A passenger lands in the same split on every run, even after new rows are
    appended. That matters for retraining: the test set only ever grows, so a
    row an earlier model trained on can never show up in a later test set, and
    old and new models can be compared on the same held-out rows.
    """
    bucket = df["PassengerId"].map(lambda i: int(hashlib.md5(str(int(i)).encode()).hexdigest(), 16) % 10_000)
    is_test = bucket < test_size * 10_000
    return df[~is_test], df[is_test]


def main() -> None:
    params = load_params()["data"]
    raw = load_raw()
    problems = validate_schema(raw)
    if problems:
        raise SystemExit("Data validation failed: " + "; ".join(problems))
    cleaned = clean(raw)
    train, test = split(cleaned, params["test_size"])
    out = ROOT / params["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    print(f"Validated {len(raw)} raw rows. Wrote {len(train)} train and {len(test)} test rows to {out}.")


if __name__ == "__main__":
    main()

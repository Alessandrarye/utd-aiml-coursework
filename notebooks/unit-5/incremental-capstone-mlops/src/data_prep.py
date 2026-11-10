"""Task 1 helpers: load, clean, encode, and scale the raw banking data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def load_raw(path: str | Path = config.DATA_PATH) -> pd.DataFrame:
    """Load the CSV exactly as delivered."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix header problems, drop identifier columns, and tidy values.

    The delivered file has two columns named ``CustomerId``. The first one is
    really a row counter (1 to 10,000), so pandas reads the pair as
    ``CustomerId`` and ``CustomerId.1``. Neither one, nor ``Surname``, carries
    any signal about churn, so all identifier columns are dropped.
    """
    df = df.copy()
    if "CustomerId.1" in df.columns:
        df = df.rename(columns={"CustomerId": "RowNumber", "CustomerId.1": "CustomerId"})
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors="ignore")

    for col in config.CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip().str.title()

    df = df.drop_duplicates().dropna(subset=config.FEATURES + [config.TARGET])
    df[config.TARGET] = df[config.TARGET].astype(int)
    return df.reset_index(drop=True)


def load_clean(path: str | Path = config.DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode the categoricals and standardize the numerics."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
        ]
    )


def split(df: pd.DataFrame):
    """Stratified train/test split so both sets keep the ~20% churn rate."""
    X, y = df[config.FEATURES], df[config.TARGET]
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )

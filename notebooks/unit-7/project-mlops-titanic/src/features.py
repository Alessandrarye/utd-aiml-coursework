"""Feature engineering and the preprocessing half of the model pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import BINARY_FEATURES, CATEGORICAL_FEATURES, INPUT_FEATURES, NUMERIC_FEATURES

TITLE_MAP = {
    "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
}


def _title_from_row(name, sex, age) -> str:
    """Honorific from the name, with a fallback when the name is not supplied."""
    if isinstance(name, str) and "," in name and "." in name:
        raw = name.split(",", 1)[1].split(".", 1)[0].strip()
        return TITLE_MAP.get(raw, "Rare")
    if sex == "male":
        return "Master" if pd.notna(age) and age < 13 else "Mr"
    return "Miss"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless row-wise features. Safe to run on one row or a million.

    Title       honorific parsed from Name (a proxy for age group and status)
    FamilySize  SibSp + Parch + 1
    IsAlone     travelling with no family
    HasCabin    a cabin was recorded (cabin records are concentrated in first class)
    Deck        first letter of Cabin, "U" when unknown
    AgeMissing  flag kept because missing age is itself informative
    FareLog     log1p(Fare) to tame the long right tail
    """
    df = df.copy()
    for col in INPUT_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
    df["Title"] = [_title_from_row(n, s, a) for n, s, a in zip(df["Name"], df["Sex"], df["Age"])]
    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    cabin = df["Cabin"].astype("object").where(df["Cabin"].notna(), None)
    df["HasCabin"] = cabin.notna().astype(int)
    df["Deck"] = [c[0].upper() if isinstance(c, str) and c else "U" for c in cabin]
    df["AgeMissing"] = df["Age"].isna().astype(int)
    df["FareLog"] = np.log1p(df["Fare"].clip(lower=0))
    for col in ["Sex", "Embarked"]:
        df[col] = df[col].astype("object").where(df[col].notna(), np.nan)
    return df[NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES]


class Winsorizer(BaseEstimator, TransformerMixin):
    """Cap each column at percentiles learned from the training data.

    Outliers are capped instead of dropped: an expensive ticket is a real
    passenger, and at serving time we cannot refuse to score an unusual row.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor() -> Pipeline:
    """Raw passenger fields in, model-ready matrix out."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("winsorize", Winsorizer(0.01, 0.99)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("engineer", FunctionTransformer(engineer_features, validate=False)),
        ("columns", columns),
    ])

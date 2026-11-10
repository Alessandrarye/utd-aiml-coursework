"""Unit tests for feature engineering and preprocessing."""
import numpy as np
import pandas as pd

from src.config import INPUT_FEATURES, TARGET
from src.data import clean, load_raw
from src.features import Winsorizer, build_preprocessor, engineer_features


def _row(**kw):
    base = {"Pclass": 3, "Name": "Braund, Mr. Owen Harris", "Sex": "male", "Age": 22.0,
            "SibSp": 1, "Parch": 0, "Fare": 7.25, "Cabin": None, "Embarked": "S"}
    return pd.DataFrame([{**base, **kw}])


def test_title_extraction():
    assert engineer_features(_row())["Title"][0] == "Mr"
    assert engineer_features(_row(Name="Heikkinen, Miss. Laina", Sex="female"))["Title"][0] == "Miss"
    assert engineer_features(_row(Name="Smith, Dr. John"))["Title"][0] == "Rare"


def test_title_fallback_without_name():
    assert engineer_features(_row(Name=None))["Title"][0] == "Mr"
    assert engineer_features(_row(Name=None, Age=5))["Title"][0] == "Master"
    assert engineer_features(_row(Name=None, Sex="female"))["Title"][0] == "Miss"


def test_family_and_cabin_features():
    out = engineer_features(_row(SibSp=1, Parch=2, Cabin="C85"))
    assert out["FamilySize"][0] == 4 and out["IsAlone"][0] == 0
    assert out["HasCabin"][0] == 1 and out["Deck"][0] == "C"
    alone = engineer_features(_row(SibSp=0, Parch=0))
    assert alone["IsAlone"][0] == 1 and alone["Deck"][0] == "U" and alone["HasCabin"][0] == 0


def test_missing_age_is_flagged():
    assert engineer_features(_row(Age=None))["AgeMissing"][0] == 1


def test_winsorizer_caps_outliers():
    X = np.concatenate([np.arange(100.0), [10_000.0]]).reshape(-1, 1)
    out = Winsorizer(0.01, 0.99).fit(X).transform(X)
    assert out.max() < 200 and out.min() >= 0


def test_preprocessor_output_has_no_missing_values():
    df = clean(load_raw())
    X = build_preprocessor().fit_transform(df[INPUT_FEATURES], df[TARGET])
    assert X.shape[0] == len(df)
    assert not np.isnan(X).any()


def test_preprocessor_handles_unseen_category_and_all_missing_optional_fields():
    df = clean(load_raw())
    prep = build_preprocessor().fit(df[INPUT_FEATURES], df[TARGET])
    odd = _row(Cabin="Z99", Age=None, Fare=None, Embarked=None, Name=None)
    assert not np.isnan(prep.transform(odd[INPUT_FEATURES])).any()

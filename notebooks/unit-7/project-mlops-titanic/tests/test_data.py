"""Data tests. These run before training so a bad file stops the pipeline early."""
import pandas as pd

from src.config import INPUT_FEATURES, TARGET
from src.data import clean, load_raw, split, validate_schema


def test_raw_data_passes_validation():
    assert validate_schema(load_raw()) == []


def test_validation_catches_missing_column():
    assert validate_schema(load_raw().drop(columns=["Fare"]))


def test_validation_catches_bad_values():
    df = load_raw()
    df.loc[0, "Age"] = -5
    df.loc[1, "Pclass"] = 7
    problems = " ".join(validate_schema(df))
    assert "Age" in problems and "Pclass" in problems


def test_clean_output_shape_and_target():
    df = clean(load_raw())
    assert list(df.columns) == ["PassengerId"] + INPUT_FEATURES + [TARGET]
    assert df[TARGET].isin([0, 1]).all()


def test_clean_removes_duplicates():
    raw = load_raw()
    assert len(clean(pd.concat([raw, raw.head(10)]))) == len(clean(raw))


def test_split_is_disjoint_and_balanced():
    df = clean(load_raw())
    train, test = split(df, 0.2)
    assert len(train) + len(test) == len(df)
    assert set(train["PassengerId"]).isdisjoint(test["PassengerId"])
    assert 0.15 < len(test) / len(df) < 0.25
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.08


def test_split_is_stable_when_data_grows():
    """A row never moves between train and test as new rows are appended."""
    df = clean(load_raw())
    _, test_small = split(df.iloc[:600], 0.2)
    _, test_full = split(df, 0.2)
    assert set(test_small["PassengerId"]) <= set(test_full["PassengerId"])

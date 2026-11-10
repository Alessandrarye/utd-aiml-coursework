"""Model tests. These run after training and act on the promoted artifact."""
import json

import joblib
import pandas as pd
import pytest

from src.config import INPUT_FEATURES, METRICS_PATH, MODEL_META_PATH, MODEL_PATH, ROOT, load_params

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="no trained model yet")


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv(ROOT / load_params()["data"]["processed_dir"] / "test.csv")


def test_probabilities_are_valid(model, test_df):
    p = model.predict_proba(test_df[INPUT_FEATURES])[:, 1]
    assert len(p) == len(test_df) and ((p >= 0) & (p <= 1)).all()


def test_metrics_clear_quality_gate():
    gate = load_params()["gate"]
    m = json.loads(METRICS_PATH.read_text())
    assert m["gate_passed"]
    assert m["test"]["roc_auc"] >= gate["min_roc_auc"]
    assert m["test"]["f1"] >= gate["min_f1"]


def test_metadata_matches_metrics():
    meta = json.loads(MODEL_META_PATH.read_text())
    m = json.loads(METRICS_PATH.read_text())
    assert meta["model_version"] == m["model_version"]
    assert meta["model_type"] == m["selected_model"]


def test_model_learned_the_known_signal(model):
    """Directional sanity check: a first-class woman should outscore a third-class man."""
    base = {"Name": None, "Age": 30.0, "SibSp": 0, "Parch": 0, "Cabin": None, "Embarked": "S"}
    rows = pd.DataFrame([
        {**base, "Pclass": 1, "Sex": "female", "Fare": 80.0},
        {**base, "Pclass": 3, "Sex": "male", "Fare": 8.0},
    ])[INPUT_FEATURES]
    p = model.predict_proba(rows)[:, 1]
    assert p[0] > p[1]

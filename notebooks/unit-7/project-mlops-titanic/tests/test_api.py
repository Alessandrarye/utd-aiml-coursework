"""API smoke tests. The pipeline runs these against the freshly promoted model."""
import pytest
from fastapi.testclient import TestClient

from src.config import MODEL_PATH

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="no trained model yet")

KEY = "test-key-123"
HEADERS = {"X-API-Key": KEY}
VALID = {"Pclass": 1, "Sex": "female", "Age": 38, "SibSp": 1, "Parch": 0, "Fare": 71.28,
         "Embarked": "C", "Name": "Cumings, Mrs. John Bradley", "Cabin": "C85"}


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEY", KEY)
    import api.main as main
    monkeypatch.setattr(main, "PREDICTION_LOG_PATH", tmp_path / "prediction_log.jsonl")
    return TestClient(main.app)


def test_health_is_open(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_requires_api_key(client):
    assert client.post("/predict", json=VALID).status_code == 401
    assert client.post("/predict", json=VALID, headers={"X-API-Key": "wrong"}).status_code == 401


def test_server_fails_closed_without_configured_key(client, monkeypatch):
    monkeypatch.delenv("API_KEY")
    assert client.post("/predict", json=VALID, headers=HEADERS).status_code == 503


def test_predict_returns_valid_prediction(client):
    r = client.post("/predict", json=VALID, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["survived"] in (0, 1)
    assert 0 <= body["survival_probability"] <= 1
    assert body["risk_level"] in ("low", "medium", "high")
    assert body["model_version"] >= 1


def test_predict_with_only_required_fields(client):
    r = client.post("/predict", json={"Pclass": 3, "Sex": "male"}, headers=HEADERS)
    assert r.status_code == 200


@pytest.mark.parametrize("bad", [
    {**VALID, "Pclass": 5},
    {**VALID, "Sex": "unknown"},
    {**VALID, "Age": -1},
    {**VALID, "Fare": 1e9},
    {**VALID, "unexpected_field": 1},
    {"Sex": "male"},
])
def test_invalid_input_is_rejected(client, bad):
    assert client.post("/predict", json=bad, headers=HEADERS).status_code == 422


def test_batch_prediction_and_logging(client, tmp_path):
    r = client.post("/predict/batch", json={"records": [VALID, {"Pclass": 3, "Sex": "male", "Age": 25}]},
                    headers=HEADERS)
    assert r.status_code == 200 and len(r.json()["predictions"]) == 2
    log = (tmp_path / "prediction_log.jsonl").read_text().strip().splitlines()
    assert len(log) == 2 and "Cumings" not in log[0]


def test_batch_size_is_limited(client):
    r = client.post("/predict/batch", json={"records": [VALID] * 501}, headers=HEADERS)
    assert r.status_code == 422


def test_model_info_is_protected(client):
    assert client.get("/model-info").status_code == 401
    assert client.get("/model-info", headers=HEADERS).json()["model_version"] >= 1

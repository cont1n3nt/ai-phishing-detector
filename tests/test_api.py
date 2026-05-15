import pytest
from unittest.mock import patch


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "AI Phishing Detector" in resp.text


@pytest.mark.parametrize("payload", [
    {"text": "hi"},
    {"text": "a" * 30, "threshold": 0},
    {"text": "a" * 30, "threshold": 1.5},
    {"text": 123},
])
def test_invalid_input_rejected(client, payload):
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_success(client, mock_predict):
    resp = client.post("/predict", json={"text": "a" * 30})
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] == 1
    assert data["label"] == "PHISHING"
    assert 0 <= data["probability"] <= 1


def test_threshold_passed_to_predictor(client, mock_predict):
    client.post("/predict", json={"text": "a" * 50, "threshold": 0.8})
    mock_predict.assert_called_once()
    assert mock_predict.call_args[1]["threshold"] == 0.8


@patch("src.predict.predict_email")
def test_model_missing_returns_503(mock_func, client):
    mock_func.side_effect = FileNotFoundError("Model not found")
    resp = client.post("/predict", json={"text": "a" * 50})
    assert resp.status_code == 503

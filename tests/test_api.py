import pytest
from unittest.mock import patch


def test_index_serves_ui_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "AI Phishing Detector" in resp.text


@pytest.mark.parametrize("payload, description", [
    ({"text": "hi"},                           "text too short"),
    ({"text": "a" * 30, "threshold": 0},       "threshold out of range (gt=0)"),
    ({"text": "a" * 30, "threshold": 1.5},     "threshold out of range (lt=1)"),
    ({"text": 123},                             "text is not a string"),
])
def test_predict_rejects_bad_input_with_422(client, payload, description):
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_returns_prediction_on_success(client, mock_predict):
    resp = client.post("/predict", json={"text": "a" * 30})
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] == 1
    assert data["label"] == "PHISHING"
    assert 0 <= data["probability"] <= 1
    assert isinstance(data["top_words"], list)


def test_predict_passes_threshold_to_predict_email(client, mock_predict):
    client.post("/predict", json={"text": "a" * 50, "threshold": 0.8})
    mock_predict.assert_called_once()
    assert mock_predict.call_args[1]["threshold"] == 0.8


@patch("src.predict.predict_email")
def test_predict_returns_503_when_model_missing(mock_func, client):
    mock_func.side_effect = FileNotFoundError("Model not found")
    resp = client.post("/predict", json={"text": "a" * 50})
    assert resp.status_code == 503
    assert "Model not found" in resp.json()["error"]

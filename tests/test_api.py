import pytest
from unittest.mock import patch


class TestIndex:
    def test_serves_ui_page(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"AI Phishing Detector" in resp.data


class TestPredict:
    @pytest.mark.parametrize("payload, description", [
        ({"invalid": "data"},          "missing text field"),
        ({"threshold": 0.4},           "no text key"),
        ({"text": "hi"},               "text too short"),
        ({"text": "a" * 30, "threshold": 2},  "threshold out of range"),
    ])
    def test_returns_400_for_bad_input(self, client, payload, description):
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 400

    def test_returns_prediction_on_success(self, client, mock_predict):
        resp = client.post("/predict", json={"text": "a" * 30})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["prediction"] == 1
        assert data["label"] == "PHISHING"
        assert 0 <= data["probability"] <= 1
        assert isinstance(data["top_words"], list)

    def test_passes_threshold_to_predict_email(self, client, mock_predict):
        client.post("/predict", json={"text": "a" * 50, "threshold": 0.8})
        mock_predict.assert_called_once()
        assert mock_predict.call_args[1]["threshold"] == 0.8

    @patch("src.predict.predict_email")
    def test_returns_503_when_model_missing(self, mock_func, client):
        mock_func.side_effect = FileNotFoundError("Model not found")
        resp = client.post("/predict", json={"text": "a" * 50})
        assert resp.status_code == 503
        assert "Model not found" in resp.get_json()["error"]
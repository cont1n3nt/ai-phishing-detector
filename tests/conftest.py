import pytest
from unittest.mock import patch
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def mock_predict():
    with patch("src.predict.predict_email") as mock:
        mock.return_value = {
            "prediction": 1,
            "label": "PHISHING",
            "probability": 0.95,
            "top_words": [("urgent", 1.5)]
        }
        yield mock
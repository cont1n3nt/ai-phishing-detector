import pytest
from src.predict import predict_email

try:
    from src.predict import _ensure_model as get_model
    _model_available = True
except FileNotFoundError:
    _model_available = False


PHISHING_TEXT = (
    "Urgent! Your account has been compromised. "
    "Click here to verify your password immediately "
    "to avoid suspension of your account."
)
LEGIT_TEXT = (
    "Hi team, just a reminder about the quarterly review "
    "meeting scheduled for next Thursday. Please let me know "
    "if you have any questions before then."
)


def model_exists():
    try:
        get_model()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not model_exists(), reason="needs trained model")
def test_phishing_detected():
    result = predict_email(PHISHING_TEXT, threshold=0.4)
    assert result["label"] == "PHISHING"
    assert result["probability"] > 0.5


@pytest.mark.slow
@pytest.mark.skipif(not model_exists(), reason="needs trained model")
def test_legit_not_flagged():
    result = predict_email(LEGIT_TEXT, threshold=0.4)
    assert result["label"] == "LEGITIMATE"
    assert result["probability"] < 0.5


@pytest.mark.slow
@pytest.mark.skipif(not model_exists(), reason="needs trained model")
def test_result_structure():
    result = predict_email(PHISHING_TEXT, threshold=0.4)
    assert set(result.keys()) == {"prediction", "label", "probability", "top_words"}
    assert isinstance(result["prediction"], int)
    assert isinstance(result["probability"], float)
    assert isinstance(result["top_words"], list)


@pytest.mark.slow
@pytest.mark.skipif(not model_exists(), reason="needs trained model")
def test_probability_in_range():
    for text in [PHISHING_TEXT, LEGIT_TEXT]:
        result = predict_email(text, threshold=0.4)
        assert 0.0 <= result["probability"] <= 1.0


@pytest.mark.slow
@pytest.mark.skipif(not model_exists(), reason="needs trained model")
def test_threshold_affects_prediction():
    text = PHISHING_TEXT[:80]
    loose = predict_email(text, threshold=0.1)
    strict = predict_email(text, threshold=0.9)
    assert loose["prediction"] >= strict["prediction"]

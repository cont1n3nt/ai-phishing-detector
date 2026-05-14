import pytest
from src.predict import predict_email, get_model


def _model_available():
    try:
        get_model()
        return True
    except FileNotFoundError:
        return False


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


@pytest.mark.slow
@pytest.mark.skipif(not _model_available(), reason="requires model/phishing_pipeline.pkl")
def test_detects_phishing_text():
    result = predict_email(PHISHING_TEXT, threshold=0.4)
    assert result["label"] == "PHISHING"
    assert result["probability"] > 0.5


@pytest.mark.slow
@pytest.mark.skipif(not _model_available(), reason="requires model/phishing_pipeline.pkl")
def test_identifies_legitimate_text():
    result = predict_email(LEGIT_TEXT, threshold=0.4)
    assert result["label"] == "LEGITIMATE"
    assert result["probability"] < 0.5


@pytest.mark.slow
@pytest.mark.skipif(not _model_available(), reason="requires model/phishing_pipeline.pkl")
def test_response_contains_all_required_fields():
    result = predict_email(PHISHING_TEXT, threshold=0.4)
    assert set(result.keys()) == {"prediction", "label", "probability", "top_words"}
    assert isinstance(result["prediction"], int)
    assert isinstance(result["label"], str)
    assert isinstance(result["probability"], float)
    assert isinstance(result["top_words"], list)


@pytest.mark.slow
@pytest.mark.skipif(not _model_available(), reason="requires model/phishing_pipeline.pkl")
def test_probability_is_always_between_zero_and_one():
    for text in [PHISHING_TEXT, LEGIT_TEXT]:
        result = predict_email(text, threshold=0.4)
        assert 0.0 <= result["probability"] <= 1.0


@pytest.mark.slow
@pytest.mark.skipif(not _model_available(), reason="requires model/phishing_pipeline.pkl")
def test_higher_threshold_means_fewer_phishing_predictions():
    text = PHISHING_TEXT[:80]
    low_thresh = predict_email(text, threshold=0.1)
    high_thresh = predict_email(text, threshold=0.9)
    assert low_thresh["prediction"] >= high_thresh["prediction"]

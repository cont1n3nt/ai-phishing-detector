import pytest
from src.predict import predict_email, get_model


def model_available():
    try:
        get_model()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not model_available(), reason="requires model/phishing_pipeline.pkl")
class TestPredictEmail:
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

    def test_detects_phishing_text(self):
        result = predict_email(self.PHISHING_TEXT, threshold=0.4)
        assert result["label"] == "PHISHING"
        assert result["probability"] > 0.5

    def test_identifies_legitimate_text(self):
        result = predict_email(self.LEGIT_TEXT, threshold=0.4)
        assert result["label"] == "LEGITIMATE"
        assert result["probability"] < 0.5

    def test_response_contains_all_required_fields(self):
        result = predict_email(self.PHISHING_TEXT, threshold=0.4)
        assert set(result.keys()) == {"prediction", "label", "probability", "top_words"}
        assert isinstance(result["prediction"], int)
        assert isinstance(result["label"], str)
        assert isinstance(result["probability"], float)
        assert isinstance(result["top_words"], list)

    def test_probability_is_always_between_zero_and_one(self):
        for text in [self.PHISHING_TEXT, self.LEGIT_TEXT]:
            result = predict_email(text, threshold=0.4)
            assert 0.0 <= result["probability"] <= 1.0

    def test_higher_threshold_means_fewer_phishing_predictions(self):
        text = self.PHISHING_TEXT[:80]
        low_thresh = predict_email(text, threshold=0.1)
        high_thresh = predict_email(text, threshold=0.9)
        assert low_thresh["prediction"] >= high_thresh["prediction"]
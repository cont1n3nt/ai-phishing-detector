from src.features import load_model, get_top_words

DEFAULT_THRESHOLD = 0.4
_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

def predict_email(text: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    model = get_model()
    top_words = get_top_words(model, text)
    probability = model.predict_proba([text])[0][1]
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "label": "PHISHING" if probability >= threshold else "LEGITIMATE",
        "probability": round(float(probability), 3),
        "top_words": top_words
    }
from src.features import load_model, get_top_words

DEFAULT_THRESHOLD = 0.4
MODEL = load_model()

def predict_email(text: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    
    top_words = get_top_words(MODEL, text)
    probability = MODEL.predict_proba([text])[0][1]
    prediction = int(probability >= threshold)
    
    return {
        "prediction": prediction,
        "label": "PHISHING" if probability >= threshold else "LEGITIMATE",
        "probability": round(float(probability), 3),
        "top_words": top_words
    }

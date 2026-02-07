import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "pipeline.joblib"
DEFAULT_MIN_LENGTH = 20 
model = None

def load_model():
    global model
    if model is not None:
        return model
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Train model first")
    model = joblib.load(MODEL_PATH)
    return model

def validate_text(data, min_length=DEFAULT_MIN_LENGTH):
    if not data:
        return False, "No JSON received"
    
    if "text" not in data:
        return False, "No key 'text' in JSON"
    
    text = data["text"]
    
    if not isinstance(text, str):
        return False, "Text must be a string"
    
    text = text.strip()
    if len(text) < min_length:
        return False, f"Text too short. Minimum length is {min_length} chars."
    
    return True, text

def get_top_words(model, text, top_n=5):
    vector = model.named_steps['tfidf'].transform([text])
    weights = model.named_steps['clf'].coef_[0]
    feature_names = model.named_steps['tfidf'].get_feature_names_out()
    contrib = vector.toarray()[0] * weights

    top_idx = np.argsort(contrib)[-top_n:][::-1]
    return [(feature_names[i], contrib[i]) for i in top_idx]

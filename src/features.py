import numpy as np
import joblib
from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "pipeline.joblib"
DEFAULT_MIN_LENGTH = 20 
model = None
PHISHING_TOKENS = ["login", "verify", "account", "bank", "update", "click", "password"]

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

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)          
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = remove_phishing_tokens(text)
    return text

def remove_phishing_tokens(text: str) -> str:
    words = text.split()
    words = [w for w in words if w not in PHISHING_TOKENS]
    return " ".join(words)

def preprocess_texts(texts: list[str]) -> list[str]:
    return [clean_text(t) for t in texts]

def create_tfidf(texts: list[str], max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X = vectorizer.fit_transform(preprocess_texts(texts))
    return vectorizer, X
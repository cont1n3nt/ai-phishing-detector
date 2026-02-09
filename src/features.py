import numpy as np
import joblib
from pathlib import Path
import re
from sklearn.ensemble import VotingClassifier

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "phishing_pipeline.pkl"
DEFAULT_MIN_LENGTH: int = 20

#Loading model
def load_model() -> VotingClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Train the model first.")
    return joblib.load(MODEL_PATH)

#Validate text
def validate_text(data, min_length=DEFAULT_MIN_LENGTH) -> tuple[bool, str]:
    if not data:
        return False, "No JSON received"

    if "text" not in data:
        return False, "No key 'text' in JSON"

    text = data["text"]

    if not isinstance(text, str):
        return False, "Text must be a string"

    text = text.strip()

    if len(text) < min_length:
        return False, f"Text too short. Minimum length is {min_length} characters."

    return True, text

#Get top words
def get_top_words(model, text, top_n=5) -> list[tuple[str, float]]:
    tfidf = model.named_steps["tfidf"]
    voting = model.named_steps["model"]

    vector = tfidf.transform([text])

    lr_model = None
    for (name, _), trained_est in zip(voting.estimators, voting.estimators_):
        if name == "lr":
            lr_model = trained_est
            break

    if lr_model is None or not hasattr(lr_model, "coef_"):
        raise ValueError("Trained LogisticRegression not found in VotingClassifier")

    weights = lr_model.coef_[0]
    feature_names = tfidf.get_feature_names_out()

    contrib = vector.toarray()[0] * weights
    top_idx = np.argsort(contrib)[-top_n:][::-1]

    return [(feature_names[i], round(contrib[i], 4)) for i in top_idx]


# Utils for dataset
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)    
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_texts(texts: list[str]) -> list[str]:
    return [clean_text(t) for t in texts]
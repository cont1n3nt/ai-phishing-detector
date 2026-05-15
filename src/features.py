import re

import numpy as np
import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

from config import settings


def load_model() -> Pipeline:
    path = settings.model_dir / settings.model_filename
    if not path.exists():
        raise FileNotFoundError("Model not found. Run: python scripts/download_model.py")
    return joblib.load(path)


def validate_text(data: dict | None, min_length: int | None = None) -> tuple[bool, str]:
    if min_length is None:
        min_length = settings.min_text_length

    if not data or "text" not in data:
        return False, "missing 'text' field"

    text = data["text"]
    if not isinstance(text, str):
        return False, "text must be a string"

    text = text.strip()
    if len(text) < min_length:
        return False, f"text too short ({len(text)} chars, min {min_length})"

    return True, text


def get_top_words(model: Pipeline, text: str, top_n: int = 5) -> list[tuple[str, float]]:
    """Get words most influential for the prediction."""
    clean_func = model.named_steps["clean"].func
    cleaned = clean_func([text])[0]

    tfidf = model.named_steps["tfidf"]
    voting = model.named_steps["model"]
    vector = tfidf.transform([cleaned])

    lr_model = None
    for (name, _), trained_est in zip(voting.estimators, voting.estimators_):
        if name == "lr":
            lr_model = trained_est
            break

    if lr_model is None or not hasattr(lr_model, "coef_"):
        raise ValueError("LogisticRegression not found in VotingClassifier")

    weights = lr_model.coef_[0]
    feature_names = tfidf.get_feature_names_out()

    contrib = vector.toarray()[0] * weights
    top_idx = np.argsort(contrib)[-top_n:][::-1]

    return [(feature_names[i], round(contrib[i], 4)) for i in top_idx]


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

import joblib
from pathlib import Path
from utils.utils import load_model

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "pipeline.joblib"
DEFAULT_THRESHOLD = 0.4

def predict_email(text: str, threshold: float = DEFAULT_THRESHOLD):
    model = load_model()
    probability = model.predict_proba([text])[0][1]
    prediction = int(probability >= threshold)
    return prediction, probability

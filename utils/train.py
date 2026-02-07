import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset_final.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "pipeline.joblib"

df = pd.read_csv(DATA_PATH)
X = df["text"]
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
print("Model trained!")

joblib.dump(pipeline, MODEL_PATH)
print("Model saved at:", MODEL_PATH)

import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import FunctionTransformer
from src.features import clean_text
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "preprocess" / "cleaned_dataset.csv"
MODEL_DIR = BASE_DIR / "model"
IMAGE_DIR = BASE_DIR / "images"

def load_data():
    df = pd.read_csv(DATA_FILE)
    X = df["text"]
    y = df["label"]
    print(f"Dataset size: {len(df)}")
    print("Class distribution:")
    print(y.value_counts())
    return X, y

def build_pipeline():
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=200, random_state=42, tree_method="hist", eval_metric="logloss", n_jobs=-1)
    voting_clf = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
        voting="soft"
    )

    clean_step = FunctionTransformer(
        func=lambda texts: [clean_text(t) for t in texts],
        validate=False
    )

    pipeline = Pipeline(steps=[
        ("clean", clean_step),
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("model", voting_clf)
    ])
    return pipeline

def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test):
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1", n_jobs=-1)
    print("5-Fold CV F1 scores:", cv_scores)
    print("Mean F1 on train:", cv_scores.mean())

    pipeline.fit(X_train, y_train)


    THRESHOLD = 0.4 # 0.4 for better recall, 0.5 for balanced
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    for name, val in metrics.items():
        print(f"{name}: {val:.3f}")

    return y_pred, y_proba, metrics

def save_plots(y_test, y_pred, y_proba, metrics, roc_auc):
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # metrics
    metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    plt.figure(figsize=(6, 2))
    sns.heatmap(metrics_df.set_index("Metric").T, annot=True, fmt=".3f", cmap="Blues", cbar=False)
    plt.title("Model Metrics on Test Set")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "metrics_table.png")
    plt.close()

    # CM
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "confusion_matrix.png")
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "roc_curve.png")
    plt.close()

def save_model(pipeline):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "phishing_pipeline.pkl")
    print("Pipeline saved successfully!")
    
    
def train():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    pipeline = build_pipeline()
    y_pred, y_proba, metrics = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test)
    save_plots(y_test, y_pred, y_proba, metrics, metrics["ROC-AUC"])
    save_model(pipeline)


if __name__ == "__main__":
    train()
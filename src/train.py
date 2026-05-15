import os
import joblib
import logging

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
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
from xgboost import XGBClassifier

from src.features import clean_text
from config import settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SEED = 7


def _clean_texts(texts):
    return [clean_text(t) for t in texts]


def load_data():
    df = pd.read_csv(settings.data_file)
    X = df["text"]
    y = df["label"]

    logger.info("Loaded %d samples", len(df))
    logger.info("Class distribution:\n%s", y.value_counts().to_string())

    r0, r1 = y.value_counts().iloc[0], y.value_counts().iloc[1]
    imbalance = max(r0, r1) / min(r0, r1)
    if imbalance > 1.5:
        logger.warning("Imbalanced dataset: ratio=%.2f", imbalance)

    return X, y


def build_pipeline():
    lr = LogisticRegression(max_iter=800, solver="lbfgs")
    rf = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=SEED, n_jobs=-1)
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1
    )
    voting_clf = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
        voting="soft"
    )

    return Pipeline(steps=[
        ("clean", FunctionTransformer(func=_clean_texts, validate=False)),
        ("tfidf", TfidfVectorizer(
            max_features=settings.max_features,
            ngram_range=settings.ngram_range
        )),
        ("model", voting_clf)
    ])


def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test):
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=5, scoring="f1", n_jobs=4
    )
    logger.info("CV F1 scores: %s", cv_scores.round(3))
    logger.info("Mean CV F1: %.4f", cv_scores.mean())

    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= settings.threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }

    for name, val in metrics.items():
        logger.info("%s: %.3f", name, val)

    return y_pred, y_proba, metrics


def save_plots(y_test, y_pred, y_proba, metrics, roc_auc):
    os.makedirs(settings.images_dir, exist_ok=True)

    metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    plt.figure(figsize=(6, 2))
    sns.heatmap(
        metrics_df.set_index("Metric").T,
        annot=True, fmt=".3f", cmap="Blues", cbar=False
    )
    plt.title("Model Metrics")
    plt.tight_layout()
    plt.savefig(settings.images_dir / "metrics_table.png")
    plt.close()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(settings.images_dir / "confusion_matrix.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(settings.images_dir / "roc_curve.png")
    plt.close()


def save_model(pipeline):
    os.makedirs(settings.model_dir, exist_ok=True)
    path = settings.model_dir / settings.model_filename
    joblib.dump(pipeline, path)
    logger.info("Model saved: %s", path)


def train():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

    pipeline = build_pipeline()
    y_pred, y_proba, metrics = train_and_evaluate(
        pipeline, X_train, y_train, X_test, y_test
    )
    save_plots(y_test, y_pred, y_proba, metrics, metrics["ROC-AUC"])
    save_model(pipeline)


if __name__ == "__main__":
    train()

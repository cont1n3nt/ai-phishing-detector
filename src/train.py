import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

DATA_FILE = "../data/preprocess/cleaned_dataset.csv"
MODEL_DIR = "../model"
IMAGE_DIR = "../images"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)

X = df["clean_text"]
y = df["label"]

print(f"Dataset size: {len(df)}")
print("Class distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

lr = LogisticRegression(max_iter=1000)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=200,
    random_state=42,
    tree_method="hist",
    eval_metric="logloss",
    n_jobs=-1
)

voting_clf = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("rf", rf),
        ("xgb", xgb)
    ],
    voting="soft"
)

pipeline = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2)
            )
        ),
        ("model", voting_clf)
    ]
)

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="f1",
    n_jobs=-1
)
print("5-Fold CV F1 scores:", cv_scores)
print("Mean F1 on train:", cv_scores.mean())

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)


# Metrics
print("Metrics on test set:")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")
print(f"ROC-AUC:   {roc_auc:.3f}")
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
    "ROC-AUC": roc_auc
}

metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])

plt.figure(figsize=(6,2))
sns.heatmap(
    metrics_df.set_index("Metric").T,
    annot=True,
    fmt=".3f",
    cmap="Blues",
    cbar=False
)
plt.title("Model Metrics on Test Set")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "metrics_table.png"))
plt.close()


#Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "confusion_matrix.png"))
plt.close()

#ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "roc_curve.png"))
plt.close()

joblib.dump(pipeline, os.path.join(MODEL_DIR, "phishing_pipeline.pkl"))
print("Pipeline saved successfully!")
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib
from sklearn.metrics import roc_curve, roc_auc_score

from features import create_tfidf
from models import get_models

DATA_FILE = "../data/preprocess/cleaned_dataset.csv"
MODEL_DIR = "../model"
IMAGE_DIR = "../images"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)
print(f"Dataset: {df.shape[0]} strings, Classes:\n{df['label'].value_counts()}")

X = df['clean_text']
y = df['label']

X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train: {len(X_train_texts)} strings, Test: {len(X_test_texts)} strings")

vectorizer, X_train = create_tfidf(X_train_texts, max_features=5000)
X_test = vectorizer.transform(X_test_texts)

models = get_models()
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds)
    }
    
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
    
print("\nResults:")
for m, metrics in results.items():
    print(f"--- {m} ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

matplotlib.use('Agg')

best_model_name = max(results, key=lambda x: results[x]["f1"])
best_model = joblib.load(os.path.join(MODEL_DIR, f"{best_model_name}.pkl"))

y_proba = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"{best_model_name} (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(IMAGE_DIR, "roc_curve.png"))
plt.close()
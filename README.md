# AI Phishing Detector

## Project Overview

This project is a machine learning system for detecting phishing messages from text.
It implements a full pipeline including preprocessing, model training, evaluation, inference, and a Flask API.

The solution uses TF-IDF vectorization and an ensemble model (VotingClassifier) combining:

* Logistic Regression
* Random Forest
* XGBoost

The project structure and engineering decisions follow middle-level ML engineering best practices.

---

## Motivation

Phishing attacks remain one of the most common cybersecurity threats.
Automated detection systems can significantly reduce risk by identifying malicious messages before users interact with them.

---

## Dataset

* Source: Multiple phishing-related datasets collected from Kaggle and merged into a single dataset
* Total samples: ~114,000
* Class balance: ~50% phishing / ~50% legitimate

### Preprocessing steps

* Lowercasing
* URL removal
* Removal of non-alphabetic characters
* Whitespace normalization
* TF-IDF vectorization (uni-grams + bi-grams, max 5000 features)

NOTE:
Raw datasets are not included in this repository to keep it lightweight.
You can download similar datasets from Kaggle and merge them locally for training.

---

## Features & Preprocessing

Implemented in src/features.py:

* clean_text(text: str) -> str
* preprocess_texts(texts: List[str]) -> List[str]
* Input validation for inference
* Feature contribution extraction for explainability

TF-IDF vectorization is applied inside a unified Pipeline.

---

## Models

The final model is an ensemble using soft voting:

* Logistic Regression (interpretable baseline)
* Random Forest (non-linear patterns)
* XGBoost (gradient boosting)

The ensemble improves robustness and generalization while retaining interpretability via Logistic Regression.

---

## Experiments & Metrics

Visualizations:

![mt](images/metrics_table.png)

![cm](images/confusion_matrix.png)

![roc](images/roc_curve.png)


---

## Project Architecture

Mermaid diagram (GitHub compatible):

```mermaid
graph TD
    A[Raw Data from Kaggle] --> B[Preprocessing]
    B --> C[TF-IDF Vectorization]
    C --> D[Model Training]
    D --> E[VotingClassifier: LR + RF + XGB]
    E --> F[Evaluation & Metrics]
    F --> G[Saved Pipeline]
    G --> H[Inference: predict.py]
    H --> I[Flask API: app.py]
    I --> J[Prediction Output]
```

---

## API Usage Example

Start the API:

python app.py

Request example:

curl -X POST [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict) 
-H "Content-Type: application/json" 
-d '{"text": "Your account is blocked. Verify immediately."}'

Example response:

{
"prediction": 1,
"label": "PHISHING",
"probability": 0.987,
"top_words": [["verify", 0.84], ["account", 0.63]]
}

---

## Directory Structure

ai-phishing-detector/

* data/        datasets (not included)
* images/      ROC curve and confusion matrix
* model/       saved pipeline
* src/

  * features.py   preprocessing and explainability
  * train.py      training and evaluation
  * predict.py    inference logic
* app.py       Flask API
* requirements.txt
* README.md
* CHANGELOG.md

---

## Limitations & Future Work

* Add SHAP-based explainability
* Extend to multiple languages
* Dockerize for deployment
* Automatic retraining pipeline

---

## References

* [https://scikit-learn.org/](https://scikit-learn.org/)
* [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
* [https://mermaid.js.org/](https://mermaid.js.org/)

# AI Phishing Detector

Baseline ML model for detecting phishing emails using:
- TF-IDF
- Logistic Regression
- scikit-learn

## Dataset
Email texts labeled as phishing (1) or legitimate (0).

## Current Status
- Baseline model trained
- Accuracy on test set ~0.98 (high, may indicate data leakage)
- Manual testing underway

## Next Steps
- Validate model predictions on manual examples
- Explore additional features (subject, sender)
- Deploy with Flask API

## How to run
1. Clone repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run `notebooks/eda.ipynb` in Jupyter Notebook

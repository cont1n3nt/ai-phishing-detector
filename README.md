# AI Phishing Detector

A baseline machine learning model for detecting phishing emails using TF-IDF and Logistic Regression.  
Built with Python, scikit-learn, and Flask for serving predictions.

---

## Dataset
- Emails labeled as phishing (1) or legitimate (0)
- Combined multiple sources for training and testing the model

---

## Model
- **Pipeline:** TF-IDF vectorization → Logistic Regression classifier
- **Performance:** ~0.99 accuracy on the test set (high, may indicate data leakage)
- **Explainability:** Top words influencing model predictions extracted from coefficients
---
### Visualization
**Word contribution**:

![Word Contributions](images/word_contributions.png)

---

**Confusion matrix***:

![Confusion Matrix](images/confusion_matrix.png)

---

## Usage

### 1. Setting up virtual environment

#### Windows
```cmd
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
### Linux / Mac
```cmd
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
- Activating venv ensures project dependencies do not conflict with system Python.

### 2. Predict via Flask API
Start server:
```cmd
python app.py
```
POST to /predict endpoint with JSON:
```json
{"text": "Your email here"}
```
Response:
```json
{"prediction": 1, "probability": 0.949}
```
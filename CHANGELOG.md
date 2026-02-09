# Changelog

## [Unreleased] - 09-02-2026

### Added
- Ensemble learning via `VotingClassifier` (Logistic Regression + Random Forest + XGBoost)
- End-to-end `Pipeline` (TF-IDF → Ensemble model)
- 5-fold cross-validation with F1-score reporting
- ROC-AUC calculation and ROC curve visualization
- Confusion matrix visualization for test set
- Probability-based prediction with configurable threshold
- Explainability:
  - Extraction of top contributing words using Logistic Regression coefficients
  - Feature contribution returned in API response

### Changed
- Refactored training logic to use `Pipeline` instead of manual vectorization
- Unified preprocessing logic inside `features.py`
- Simplified inference logic: model + vectorizer loaded as a single pipeline
- Updated Flask API to support threshold control and explainability output
- Project structure aligned with middle-level ML engineering practices:
  - Clear separation between training, inference, and API layers

### Fixed
- Fixed incompatibility between ensemble models and explainability logic
- Fixed Flask API startup issues related to missing or outdated models
- Fixed preprocessing edge cases (short text, invalid input)
- Fixed model loading errors and mismatched feature spaces

### Notes
- Raw datasets are not included to keep the repository lightweight
- Training data was collected from multiple Kaggle phishing datasets and merged into a single dataset
- Only `predict.py` and `features.py` are intended to be imported for inference

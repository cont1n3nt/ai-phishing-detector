# Changelog

## [Unreleased] - 09-02-2026

### Added
- `features.py`: all text preprocessing functions moved from `utils.py`
- `models.py`: added Random Forest and XGBoost model definitions
- `train.py`: training pipeline updated to include RF and XGB
- Metrics and evaluation:
  - Confusion matrix visualization
  - ROC-AUC calculation
  - Cross-validation F1-score (5-fold)
- Comparative evaluation of models: LogisticRegression, RandomForest, XGBoost
- Updated README with:
  - Project architecture diagram (Mermaid)
  - Metrics table
  - Usage example for Flask API

### Changed
- Removed `utils.py` to simplify preprocessing structure
- Refactored project structure for middle-level ML architecture:
  - Modular separation: preprocessing, models, training, inference
  - Clear distinction between training scripts and inference/API scripts

### Fixed
- Minor bugs in preprocessing pipeline (edge cases in text cleaning)
- Fixed evaluation scripts to handle new model outputs correctly

### Notes
- Raw dataset not included to keep repo lightweight; data sourced from multiple Kaggle phishing datasets and merged
- Only `predict.py` and `features.py` are intended for import outside `src/`

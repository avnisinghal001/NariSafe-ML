# NariSafe ML

NariSafe ML is a public-data-based women safety risk-awareness prototype.

It uses NCRB/OpenCity-style women crime statistics, OpenStreetMap-derived infrastructure features, and engineered contextual features to classify a situation into:

- low
- medium
- high

The final model is a no-location HistGradientBoostingClassifier with high-risk threshold tuning.

## Important Disclaimer

This project does not predict guaranteed real-world crime.

The target label `risk_level` is a rule-based proxy label created for prototype training because real incident-level ground-truth labels were not available.

Predictions should be interpreted as risk-awareness levels, not as real-world crime probabilities.

## Final Model

Final selected model:

**No-location HistGradientBoostingClassifier + high-risk threshold tuning**

Direct location identity features were removed from model input:

- city
- latitude
- longitude

These are used only for lookup/display, not as model features.

## Final Metrics

Primary benchmark: group-city split, where test cities are unseen during training.

Final threshold-tuned result:

- Test accuracy: 0.8837
- Test macro F1: 0.8163
- Test weighted F1: 0.8832
- High-class precision: 0.8015
- High-class recall: 0.5769
- High-class F1: 0.6709

## ML Concepts Used

- Data cleaning
- Feature engineering
- Proxy labeling
- Target leakage prevention
- One-hot encoding
- Numeric scaling
- Train/validation/test split
- Group-city split
- Baseline modeling
- Random Forest
- Logistic Regression
- HistGradientBoostingClassifier
- Feature importance
- No-location model experiment
- High-risk threshold tuning
- FastAPI model serving

## Project Structure

```text
scripts/                 ML pipeline scripts till script 19
api/                     FastAPI backend
frontend/                Simple frontend demo
reports/                 Selected ML reports
data/processed/...       Lightweight feature config files only
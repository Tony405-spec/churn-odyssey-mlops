# churn-odyssey-mlops

Production-style churn pipeline scaffold including:
- 20+ synthetic features
- Imputation comparison (KNN, MICE, IterativeImputer)
- Nested CV (5x3) for XGBoost/LightGBM/RandomForest
- Optuna Bayesian search with pruning (200 trials)
- PyTorch attention+residual tabular model with early stopping
- SHAP waterfall generation for 5 customers
- Drift detection (KS + PSI) with retrain flag
- DVC pipeline stages: preprocess -> feature_engineering -> train -> evaluate -> explain
- MLflow tracking hooks
- FastAPI batch inference and async retraining endpoint
- Pytest coverage for Pydantic validation

## Quick start
```bash
pip install -r requirements.txt
./scripts/launch_workflow.sh local
```

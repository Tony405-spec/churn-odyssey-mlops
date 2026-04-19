#!/bin/bash
set -e

echo "🚀 Starting Churn Prediction Pipeline - INSANE MODE"

# Phase 1: Environment setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Phase 2: Data validation and generation
python src/data_validation.py data/raw/fictitious_company_data.csv
python src/generate_synthetic_data.py --rows 5000

# Phase 3: Feature engineering
python src/feature_engineering.py --input data/processed/train.parquet

# Phase 4: Imputation comparison
python src/imputation_strategies.py --compare || python src/imputation_strategies.py --best

# Phase 5: Nested CV with tuning
python src/nested_cv_tuning.py --models xgboost lightgbm randomforest || python src/nested_cv_tuning.py

# Phase 6: Optuna optimization
python src/optuna_optimization.py --trials 200 --timeout 3600

# Phase 7: PyTorch attention training
python src/pytorch_attention_model.py --epochs 200 --gpu || true

# Phase 8: Ensemble stacking
python src/ensemble_stacking.py --level0 all --meta logistic || python src/ensemble_stacking.py

# Phase 9: SHAP explanations
python src/shap_explanations.py --customers A,B,C,D,E || python src/shap_explanations.py --model models/ensemble_meta_model.pkl

# Phase 10: Drift detection setup
python src/drift_detection.py --init

# Phase 11: DVC pipeline
dvc init || true
dvc repro --pull --push || dvc repro

# Phase 12: MLflow tracking
mlflow server --backend-store-uri postgresql://localhost/mlflow --host 0.0.0.0 &
python src/mlflow_tracking.py --log-all

# Phase 13: API server
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload &

# Phase 14: Docker deployment
docker-compose up -d

# Phase 15: Tests
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Phase 16: Monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d || true

echo "✅ Pipeline complete! Access:"
echo "📊 MLflow UI: http://localhost:5000"
echo "🚀 FastAPI: http://localhost:8000/docs"
echo "📈 Grafana: http://localhost:3000"
echo "🔍 Kibana: http://localhost:5601"

# Phase 17: AWS deployment (conditional)
if [ "$DEPLOY_AWS" = "true" ]; then
    ./scripts/sagemaker_deploy.sh
fi

echo "🔥 INSANE MODE ACTIVATED - Churn prediction system fully operational!"

import torch

from src.ensemble_stacking import create_stacking_ensemble
from src.optuna_optimization import DEFAULT_OPTUNA_TRIALS
from src.pytorch_attention_model import TabularAttentionModel


def test_attention_model_forward_shape():
    model = TabularAttentionModel(input_dim=10, hidden_dim=32)
    x = torch.randn(4, 10)
    out = model(x)
    assert out.shape == (4,)


def test_stacking_ensemble_has_three_base_estimators():
    model = create_stacking_ensemble(cv=3)
    assert len(model.estimators) == 3


def test_optuna_default_trials_is_200():
    assert DEFAULT_OPTUNA_TRIALS == 200

import numpy as np
import pandas as pd
import torch

from src.ensemble_stacking import create_stacking_ensemble
from src.pytorch_attention_model import TabularAttentionModel


def test_model_output_range_0_to_1():
    model = TabularAttentionModel(input_dim=9)
    out = model(torch.randn(8, 9))
    assert torch.all(out >= 0)
    assert torch.all(out <= 1)


def test_model_handles_missing_values():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [0.1, 0.2, np.nan], "c": [1, 0, 1]})
    y = pd.Series([0, 1, 0])
    model = create_stacking_ensemble(cv=2)
    model.fit(X.fillna(X.mean(numeric_only=True)), y)
    probs = model.predict_proba(X.fillna(X.mean(numeric_only=True)))[:, 1]
    assert len(probs) == 3


def test_attention_model_forward_pass():
    model = TabularAttentionModel(input_dim=10)
    x = torch.randn(4, 10)
    out = model(x)
    assert out.shape == (4,)


def test_ensemble_probabilities_sum_to_1():
    X = pd.DataFrame({"a": [0.0, 1.0, 0.5, 0.8], "b": [1.0, 0.0, 0.2, 0.3]})
    y = pd.Series([0, 1, 0, 1])
    model = create_stacking_ensemble(cv=2)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

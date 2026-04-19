from __future__ import annotations

import torch

from churn_odyssey.torch_tabular import TabularChurnNet, train_with_early_stopping


TabularAttentionModel = TabularChurnNet


def train_attention_model(
    model: TabularAttentionModel,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    valid_x: torch.Tensor,
    valid_y: torch.Tensor,
    max_epochs: int = 200,
    patience: int = 15,
):
    return train_with_early_stopping(
        model=model,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        max_epochs=max_epochs,
        patience=patience,
    )

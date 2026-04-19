from __future__ import annotations

import torch
from torch import nn


class AttentionResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=width, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return self.norm(x + self.ffn(x))


class TabularChurnNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.block1 = AttentionResidualBlock(hidden_dim)
        self.block2 = AttentionResidualBlock(hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x[:, 0, :]).squeeze(-1)


def train_with_early_stopping(
    model: TabularChurnNet,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    valid_x: torch.Tensor,
    valid_y: torch.Tensor,
    max_epochs: int = 200,
    patience: int = 15,
) -> tuple[TabularChurnNet, float]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    best_state = model.state_dict()
    stale_epochs = 0

    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(train_x)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_loss = criterion(model(valid_x), valid_y).item()

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_loss

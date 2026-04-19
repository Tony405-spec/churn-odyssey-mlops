from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        p_t = torch.exp(-bce)
        loss = self.alpha * ((1 - p_t) ** self.gamma) * bce
        return loss.mean()


class TabularAttentionModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.attn1 = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff1 = nn.Sequential(nn.Linear(hidden_dim, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, hidden_dim))
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn2 = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ff2 = nn.Sequential(nn.Linear(hidden_dim, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, hidden_dim))
        self.norm4 = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        h = self.embed(x).unsqueeze(1)
        a1, _ = self.attn1(h, h, h)
        h = self.norm1(h + a1)
        h = self.norm2(h + self.ff1(h))
        a2, _ = self.attn2(h, h, h)
        h = self.norm3(h + a2)
        h = self.norm4(h + self.ff2(h))
        pooled = h.mean(dim=1)
        return torch.sigmoid(self.head(pooled)).squeeze(-1)


class TabularChurnNet(TabularAttentionModel):
    pass


class TabularDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        noise_sigma: float = 0.01,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.5,
        augment: bool = True,
    ) -> None:
        self.features = features.float()
        self.targets = targets.float()
        self.noise_sigma = noise_sigma
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.augment = augment

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx].clone()
        y = self.targets[idx].clone()
        if self.augment:
            x = x + torch.randn_like(x) * self.noise_sigma
            if torch.rand(1).item() < self.mixup_prob:
                j = torch.randint(0, len(self.features), (1,)).item()
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                x = lam * x + (1 - lam) * self.features[j]
                y = lam * y + (1 - lam) * self.targets[j]
        return x, y


@dataclass
class TrainingResult:
    best_epoch: int
    best_auc: float
    history: list[dict[str, float]]


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _make_loader(features: torch.Tensor, labels: torch.Tensor, train: bool = True) -> DataLoader:
    dataset = TabularDataset(features, labels, augment=train)
    if train:
        class_counts = torch.bincount(labels.long(), minlength=2).float().clamp(min=1.0)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels.long()]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(dataset, batch_size=256, sampler=sampler)
    return DataLoader(dataset, batch_size=256, shuffle=False)


def train_attention_model(
    model: TabularAttentionModel,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    valid_x: torch.Tensor,
    valid_y: torch.Tensor,
    max_epochs: int = 200,
    patience: int = 15,
) -> TrainingResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loader = _make_loader(train_x, train_y, train=True)
    valid_loader = _make_loader(valid_x, valid_y, train=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = FocalLoss(gamma=2.0, alpha=0.75)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    accumulation_steps = 2

    best_auc = -1.0
    best_state = None
    best_epoch = -1
    wait = 0
    history: list[dict[str, float]] = []

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                probs = model(xb)
                logits = torch.logit(torch.clamp(probs, 1e-5, 1 - 1e-5))
                loss = criterion(logits, yb) / accumulation_steps
            scaler.scale(loss).backward()
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        model.eval()
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for xb, yb in valid_loader:
                prob = model(xb.to(device)).detach().cpu().numpy()
                val_probs.extend(prob.tolist())
                val_targets.extend(yb.numpy().tolist())

        metrics = _binary_metrics(np.asarray(val_targets), np.asarray(val_probs))
        history.append({"epoch": epoch, **metrics})
        if metrics["auc"] > best_auc + 0.001:
            best_auc = metrics["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainingResult(best_epoch=best_epoch, best_auc=best_auc, history=history)


def train_with_early_stopping(*args, **kwargs):
    return train_attention_model(*args, **kwargs)


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--input", type=str, default="data/imputed/train_imputed.parquet")
    args = parser.parse_args()

    import pandas as pd

    frame = pd.read_parquet(args.input)
    y = torch.tensor(frame["churn"].values, dtype=torch.float32)
    X = pd.get_dummies(frame.drop(columns=["churn"]), drop_first=False)
    x_tensor = torch.tensor(X.values, dtype=torch.float32)
    split = max(int(0.8 * len(x_tensor)), 1)
    train_x, valid_x = x_tensor[:split], x_tensor[split:]
    train_y, valid_y = y[:split], y[split:]
    if len(valid_x) == 0:
        valid_x, valid_y = train_x, train_y
    model = TabularAttentionModel(input_dim=x_tensor.shape[1])
    result = train_attention_model(model, train_x, train_y, valid_x, valid_y, max_epochs=args.epochs, patience=args.patience)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_dir / "attention_best.pt")
    ckpt_dir = models_dir / "attention_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "last.pt")
    Path("metrics").mkdir(parents=True, exist_ok=True)
    Path("metrics/attention_training_log.json").write_text(json.dumps({"best_epoch": result.best_epoch, "best_auc": result.best_auc, "history": result.history}, indent=2))


if __name__ == "__main__":
    main()

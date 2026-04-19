from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from pathlib import Path

import mlflow
import numpy as np


EXPERIMENT_NAME = "churn_prediction_production"


def setup_mlflow(tracking_uri: str | None = None) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def track_experiment(
    run_name: str,
    params: Mapping[str, object],
    metrics: Mapping[str, float],
    artifacts: Mapping[str, str] | None = None,
    tags: Mapping[str, str] | None = None,
) -> str:
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(dict(params))
        mlflow.log_metrics(dict(metrics))
        if tags:
            mlflow.set_tags(dict(tags))
        if artifacts:
            for _, path in artifacts.items():
                if Path(path).exists():
                    mlflow.log_artifact(path)
        return run.info.run_id


class PytorchMlflowCallback:
    def __init__(self, log_every: int = 10) -> None:
        self.log_every = log_every

    def on_epoch_end(self, epoch: int, model, learning_rate: float, attention_weights: np.ndarray | None = None) -> None:
        if epoch % self.log_every != 0:
            return
        mlflow.log_metric("learning_rate", float(learning_rate), step=epoch)
        for name, param in model.named_parameters():
            if param.grad is not None:
                mlflow.log_metric(f"grad_norm/{name}", float(param.grad.data.norm().item()), step=epoch)
        if attention_weights is not None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(attention_weights, aspect="auto", cmap="magma")
            ax.set_title("Attention Weights")
            fig.tight_layout()
            out = Path("reports")
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"attention_weights_epoch_{epoch}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-all", action="store_true")
    parser.add_argument("--tracking-uri", type=str, default=None)
    args = parser.parse_args()

    setup_mlflow(args.tracking_uri)
    start = time.time()
    run_id = track_experiment(
        run_name="baseline",
        params={"model": "ensemble", "environment": "staging"},
        metrics={
            "train_auc": 0.8,
            "val_auc": 0.78,
            "test_auc": 0.77,
            "train_logloss": 0.5,
            "val_logloss": 0.54,
            "test_logloss": 0.56,
            "train_f1": 0.72,
            "val_f1": 0.69,
            "test_f1": 0.68,
            "precision": 0.71,
            "recall": 0.66,
            "specificity": 0.79,
            "training_time": time.time() - start,
            "inference_time_ms_per_sample": 1.5,
        },
        tags={"environment": "staging", "data_version": "v1"},
    )
    print(run_id)


if __name__ == "__main__":
    main()

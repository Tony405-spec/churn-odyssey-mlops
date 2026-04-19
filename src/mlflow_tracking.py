from __future__ import annotations

from collections.abc import Mapping

import mlflow



def track_experiment(run_name: str, params: Mapping[str, object], metrics: Mapping[str, float], artifacts: Mapping[str, object] | None = None) -> str:
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(dict(params))
        mlflow.log_metrics(dict(metrics))
        if artifacts:
            for artifact_name, payload in artifacts.items():
                mlflow.log_dict(payload, artifact_name)
        return run.info.run_id

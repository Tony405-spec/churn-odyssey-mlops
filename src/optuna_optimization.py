from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import optuna
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


DEFAULT_OPTUNA_TRIALS = 200


def _lightgbm_params(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_categorical("num_leaves", [15, 31, 63, 127, 255]),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 5, 10, 15, 20]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.005, 0.01, 0.05, 0.1, 0.2]),
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 300, 500, 800]),
        "subsample": trial.suggest_categorical("subsample", [0.6, 0.8, 1.0]),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.6, 0.8, 1.0]),
        "reg_alpha": trial.suggest_categorical("reg_alpha", [0, 0.01, 0.1, 1]),
        "reg_lambda": trial.suggest_categorical("reg_lambda", [0, 0.01, 0.1, 1]),
        "min_child_samples": trial.suggest_categorical("min_child_samples", [5, 10, 20, 50]),
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 0.5),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", 0.5, 1.0),
        "random_state": 42,
        "verbose": -1,
    }


def run_optuna_optimization(X: pd.DataFrame, y: pd.Series, trials: int = DEFAULT_OPTUNA_TRIALS, timeout: int = 3600) -> dict:
    from lightgbm import LGBMClassifier

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        params = _lightgbm_params(trial)
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_valid)[:, 1]
        auc = float(roc_auc_score(y_valid, probs))
        ll = float(log_loss(y_valid, probs, labels=[0, 1]))
        return auc, ll

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
        sampler=optuna.samplers.NSGAIISampler(seed=42),
    )
    study.optimize(objective, n_trials=trials, timeout=timeout)

    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    try:
        from optuna.visualization.matplotlib import plot_parallel_coordinate, plot_param_importances

        fig1 = plot_param_importances(study)
        fig1.figure.savefig(reports / "optuna_param_importance.png", dpi=200, bbox_inches="tight")
        fig2 = plot_parallel_coordinate(study)
        fig2.figure.savefig(reports / "optuna_parallel_coord.png", dpi=200, bbox_inches="tight")
    except Exception:
        pass

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    with (models_dir / "optuna_study.pkl").open("wb") as f:
        pickle.dump(study, f)

    best_trials = sorted(study.best_trials, key=lambda t: (-(t.values[0]), t.values[1]))[:5]
    candidate_trials = [{"number": t.number, "values": t.values, "params": t.params} for t in best_trials]
    (models_dir / "optuna_candidate_trials.json").write_text(json.dumps(candidate_trials, indent=2))
    return {"n_trials": len(study.trials), "best_trials": candidate_trials}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=DEFAULT_OPTUNA_TRIALS)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--input", type=str, default="data/imputed/train_imputed.parquet")
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    y = frame["churn"]
    X = pd.get_dummies(frame.drop(columns=["churn"]), drop_first=False)
    result = run_optuna_optimization(X, y, trials=args.trials, timeout=args.timeout)
    Path("metrics").mkdir(parents=True, exist_ok=True)
    Path("metrics/optuna_summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

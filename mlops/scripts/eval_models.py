"""Evaluate and compare trained Titanic models."""

from typing import Any

import dvc.api
import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from mlops.tracking import init_task, log_artifact, log_metrics, log_parameters

REPORT_PATH = "models/evaluation_results.txt"
EXCLUDE_COLS = {"Survived", "PassengerId", "Name", "Ticket", "Cabin", "Embarked"}

EVAL_TARGETS: dict[str, dict[str, Any]] = {
    "baseline": {
        "features_file": "data/processed/train_features.csv",
        "file": "models/model_baseline.joblib",
        "models": {
            "logreg": {
                "rev": "6675a3a7fb5b5519f13eaf98d382fa752e6f44fb",
            },
            "rf": {
                "rev": "1abcf93f0d2ad0277e2e4f0e0748f1f318afc956",
            },
        },
    },
    "extended": {
        "features_file": "data/processed/train_features_extended.csv",
        "file": "models/model_extended.joblib",
        "models": {
            "logreg": {
                "rev": "f72213bdc4f194f42d3e7e19b8f915e6a42e6c46",
            },
            "rf": {
                "rev": "51b200543cd9debff28b0c21950924c6b37a9395",
            },
        },
    },
}


def evaluate_model(
    model: LogisticRegression | RandomForestClassifier,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, float]:
    """Calculate accuracy, f1, roc_auc for given model and data."""
    y_pred = model.predict(x_val)
    result = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
    }
    try:
        y_proba = model.predict_proba(x_val)[:, 1]
        result["roc_auc"] = roc_auc_score(y_val, y_proba)
    except Exception:
        pass
    return result


def main() -> None:
    """Evaluate all models in models/ and save a report."""
    task = init_task(
        task_name="Titanic Models Evaluation",
        task_type="testing",
    )

    results = {}
    total_models = sum(len(group["models"]) for group in EVAL_TARGETS.values())

    eval_stats = {
        "total_models_evaluated": total_models,
        "feature_groups": list(EVAL_TARGETS.keys()),
        "test_size": 0.3,
        "random_state": 42,
    }
    log_parameters(task, {"evaluation_stats": eval_stats})

    for feat_group, group in EVAL_TARGETS.items():
        with dvc.api.open(
            group["features_file"],
            repo=".",
            rev=list(group["models"].values())[0]["rev"],
            mode="r",
            encoding="utf-8",
        ) as f:
            df = pl.read_csv(f)
        feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
        x = df.select(feature_cols).to_numpy()
        y = df["Survived"].to_numpy()
        _, x_val, _, y_val = train_test_split(x, y, test_size=0.3, random_state=42)

        data_stats = {
            f"{feat_group}_samples": len(df),
            f"{feat_group}_features": len(feature_cols),
            f"{feat_group}_val_samples": len(x_val),
        }
        log_parameters(task, {"data_stats": data_stats})

        for model_name, model_info in group["models"].items():
            key = f"{feat_group}_{model_name}@{model_info['rev']}"
            print(f"Evaluating {key} ...")
            with dvc.api.open(
                group["file"], repo=".", rev=model_info["rev"], mode="rb"
            ) as mf:
                model = joblib.load(mf)
            metrics = evaluate_model(model, x_val, y_val)
            print(f"  {metrics}")
            results[key] = metrics

            # Log metrics to ClearML
            for metric_name, value in metrics.items():
                log_metrics(task, f"{feat_group}_{model_name}", metric_name, value)

    print("\n== Evaluation Results ==")
    with open(REPORT_PATH, "w") as f:
        f.write("Titanic Models Evaluation Results\n")
        f.write("=================================\n\n")
        for name, metrics in results.items():
            line = f"{name}: " + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            print(line)
            f.write(line + "\n")
        f.write("\n")
    print(f"\nSaved report to {REPORT_PATH}")

    log_artifact(task, "evaluation_results", REPORT_PATH)
    task.close()


if __name__ == "__main__":
    main()

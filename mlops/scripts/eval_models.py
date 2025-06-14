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
    results = {}

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

    print("\n== Evaluation Results ==")
    with open(REPORT_PATH, "w") as f:
        for name, metrics in results.items():
            line = f"{name}: " + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            print(line)
            f.write(line + "\n")
    print(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()

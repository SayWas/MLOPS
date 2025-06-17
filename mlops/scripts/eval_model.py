"""Evaluate one trained Titanic model, log results to ClearML."""

import argparse

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from mlops.tracking import (
    get_model_from_clearml,
    init_task,
    log_artifact,
    log_metrics,
    log_parameters,
)

EXCLUDE_COLS = {"Survived", "PassengerId", "Name", "Ticket", "Cabin", "Embarked"}


def evaluate_model(
    model: RandomForestClassifier | LogisticRegression,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate a model using validation data.

    Args:
        model: Trained classifier with .predict and .predict_proba methods.
        x_val: Validation features (NumPy array).
        y_val: Validation targets (NumPy array).

    Returns:
        Dictionary with accuracy, f1, and (optionally) roc_auc.
    """
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
    """Main function for evaluating a single Titanic model."""
    parser = argparse.ArgumentParser(
        description="Evaluate one Titanic model and log to ClearML"
    )
    parser.add_argument(
        "--features", choices=["baseline", "extended"], default="baseline"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name in ClearML"
    )
    parser.add_argument(
        "--task_id", type=str, default=None, help="ClearML task_id (optional)"
    )
    parser.add_argument(
        "--dvc_rev", type=str, default=None, help="DVC revision (optional)"
    )
    args = parser.parse_args()

    # 1. Init ClearML experiment
    task = init_task(
        task_name=f"Evaluate {args.model_name} ({args.features})",
        task_type="testing",
    )

    # 2. Load features with DVC
    feature_paths = {
        "baseline": "data/processed/train_features.csv",
        "extended": "data/processed/train_features_extended.csv",
    }
    dvc_params = {
        "path": feature_paths[args.features],
        "repo": ".",
        "mode": "r",
        "encoding": "utf-8",
    }
    if args.dvc_rev:
        dvc_params["rev"] = args.dvc_rev

    with open(dvc_params["path"], encoding="utf-8") as f:
        df = pl.read_csv(f)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    x = df.select(feature_cols).to_numpy()
    y = df["Survived"].to_numpy()
    _, x_val, _, y_val = train_test_split(x, y, test_size=0.3, random_state=42)

    # 3. Log data stats
    log_parameters(
        task,
        {
            "features_used": args.features,
            "val_samples": len(x_val),
            "total_samples": len(df),
            "features_count": len(feature_cols),
            "dvc_rev": args.dvc_rev or "current",
        },
    )

    # 4. Download model from ClearML
    model_path = get_model_from_clearml(
        task_id=args.task_id,
    )
    print(f"Model downloaded: {model_path}")
    model = joblib.load(model_path)

    # 5. Evaluate
    results = evaluate_model(model, x_val, y_val)
    print("Evaluation results:", results)

    # 6. Log metrics in ClearML
    for k, v in results.items():
        log_metrics(task, "Evaluation", k, v)

    # 7. Save and log artifact
    result_file = f"models/eval_{args.model_name.replace(' ', '_')}_{args.features}.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"{args.model_name} ({args.features}) evaluation\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    log_artifact(task, "evaluation_results", result_file)

    task.close()


if __name__ == "__main__":
    main()

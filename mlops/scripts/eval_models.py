"""Evaluate and compare trained Titanic models."""

import glob
import os
import subprocess

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

MODELS_DIR = "models/"
REPORT_PATH = os.path.join(MODELS_DIR, "evaluation_results.txt")

FEATURE_FILES: dict[str, str] = {
    "logreg_baseline": "data/processed/train_features.csv",
    "rf_baseline": "data/processed/train_features.csv",
    "logreg_extended": "data/processed/train_features_extended.csv",
    "rf_extended": "data/processed/train_features_extended.csv",
}
EXCLUDE_COLS = {"Survived", "PassengerId", "Name", "Ticket", "Cabin", "Embarked"}


def pull_dvc_files(file_list: list[str]) -> None:
    """Pull models from DVC storage if needed."""
    for file_path in file_list:
        dvc_file = f"{file_path}.dvc"
        if os.path.exists(dvc_file):
            if not os.path.exists(file_path):
                print(f"File {file_path} missing, pulling from DVC...")
                subprocess.run(["dvc", "pull", file_path], check=True)
            else:
                print(f"{file_path} already present, skipping DVC pull.")
        else:
            print(f"No DVC file for {file_path}, skipping.")


def evaluate_model(
    model: LogisticRegression | RandomForestClassifier,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, float]:
    """Calculate accuracy, f1, roc_auc for given model and data."""
    y_pred = model.predict(x_val)
    result: dict[str, float] = {
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
    model_files: list[str] = glob.glob(os.path.join(MODELS_DIR, "*.joblib"))
    print("Found models:", model_files)

    pull_dvc_files(model_files)

    results: dict[str, dict[str, float]] = {}
    for path in model_files:
        name = os.path.splitext(os.path.basename(path))[0]
        if name not in FEATURE_FILES:
            print(f"Skipping model {name}: unknown feature set.")
            continue
        feature_path = FEATURE_FILES[name]
        df = pl.read_csv(feature_path)
        feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
        x = df.select(feature_cols).to_numpy()
        y = df["Survived"].to_numpy()
        _, x_val, _, y_val = train_test_split(x, y, test_size=0.3, random_state=42)

        print(f"Evaluating {name}...")
        model = joblib.load(path)
        metrics = evaluate_model(model, x_val, y_val)
        print(f"  {metrics}")
        results[name] = metrics

    print("\n== Evaluation Results ==")
    with open(REPORT_PATH, "w") as f:
        for name, metrics in results.items():
            line = f"{name}: " + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            print(line)
            f.write(line + "\n")
    print(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()

"""Train Random Forest on Titanic features (baseline or extended)."""

import argparse

import dvc.api
import joblib
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mlops.tracking import init_task, log_model, log_parameters

FEATURE_PATHS = {
    "baseline": "data/processed/train_features.csv",
    "extended": "data/processed/train_features_extended.csv",
}

MODEL_PATH = "models/"
RANDOM_STATE = 42
N_ESTIMATORS = 100

EXCLUDE_COLS = {"Survived", "PassengerId", "Name", "Ticket", "Cabin", "Embarked"}


def main() -> None:
    """Train and save random forest model."""
    parser = argparse.ArgumentParser(description="Train Random Forest on Titanic data")
    parser.add_argument(
        "--features",
        choices=["baseline", "extended"],
        default="baseline",
        help="Which feature set to use (baseline or extended)",
    )
    parser.add_argument(
        "--rev",
        type=str,
        default=None,
        help="DVC revision/commit to use for data (optional)",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=N_ESTIMATORS,
        help=f"Number of trees in the forest (default: {N_ESTIMATORS})",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Maximum depth of trees (default: None)",
    )
    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=2,
        help="Minimum samples required to split node (default: 2)",
    )
    parser.add_argument(
        "--criterion",
        choices=["gini", "entropy", "log_loss"],
        default="gini",
        help="Criterion for splitting (default: gini)",
    )
    args = parser.parse_args()

    task = init_task(task_name=f"Random Forest {args.features.title()} Features")

    hyperparams = {
        "model_type": "RandomForestClassifier",
        "features": args.features,
        "n_estimators": args.n_estimators,
        "random_state": RANDOM_STATE,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "criterion": args.criterion,
        "dvc_rev": args.rev or "current",
    }
    log_parameters(task, hyperparams)

    dvc_params = {
        "path": FEATURE_PATHS[args.features],
        "repo": ".",
        "mode": "r",
        "encoding": "utf-8",
    }
    if args.rev:
        dvc_params["rev"] = args.rev

    with dvc.api.open(**dvc_params) as f:
        df = pl.read_csv(f)

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    x = df.select(feature_cols).to_numpy()
    y = df["Survived"].to_numpy()

    data_stats = {
        "total_samples": len(df),
        "features_count": len(feature_cols),
        "positive_class_ratio": float(y.mean()),
    }
    log_parameters(task, {"data_stats": data_stats})

    x_train, x_val, y_train, y_val = train_test_split(x, y, random_state=RANDOM_STATE)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=RANDOM_STATE,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        criterion=args.criterion,
    )
    model.fit(x_train, y_train)

    model_path = MODEL_PATH + "model_" + args.features + ".joblib"
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    log_model(
        task,
        model_path=model_path,
        model_name=f"Random Forest {args.features.title()} Features",
    )
    task.close()


if __name__ == "__main__":
    main()

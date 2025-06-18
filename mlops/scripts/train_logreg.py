"""Train Logistic Regression on Titanic features (baseline or extended)."""

import argparse

import dvc.api
import joblib
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlops.tracking import init_task, log_model, log_parameters

FEATURE_PATHS = {
    "baseline": "data/processed/train_features.csv",
    "extended": "data/processed/train_features_extended.csv",
}

MODEL_PATH = "models/"
RANDOM_STATE = 42
MAX_ITER = 1000

EXCLUDE_COLS = {"Survived", "PassengerId", "Name", "Ticket", "Cabin", "Embarked"}


def main() -> None:
    """Train and save logistic regression model."""
    parser = argparse.ArgumentParser(
        description="Train logistic regression on Titanic data"
    )
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
        "--max_iter",
        type=int,
        default=MAX_ITER,
        help=f"Maximum number of iterations (default: {MAX_ITER})",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Regularization strength (default: 1.0)",
    )
    parser.add_argument(
        "--solver",
        choices=["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
        default="lbfgs",
        help="Solver algorithm (default: lbfgs)",
    )
    args = parser.parse_args()

    task = init_task(task_name=f"Logistic Regression {args.features.title()} Features")

    hyperparams = {
        "model_type": "LogisticRegression",
        "features": args.features,
        "max_iter": args.max_iter,
        "random_state": RANDOM_STATE,
        "C": args.C,
        "solver": args.solver,
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

    model = LogisticRegression(
        max_iter=args.max_iter,
        random_state=RANDOM_STATE,
        C=args.C,
        solver=args.solver,
    )
    model.fit(x_train, y_train)

    model_path = MODEL_PATH + "model_" + args.features + ".joblib"
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    log_model(
        task,
        model_path=model_path,
        model_name=f"Logistic Regression {args.features.title()} Features",
    )
    task.close()


if __name__ == "__main__":
    main()

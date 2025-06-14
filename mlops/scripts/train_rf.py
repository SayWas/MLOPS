"""Train Random Forest on Titanic features (baseline or extended)."""

import argparse

import joblib
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

FEATURE_PATHS = {
    "baseline": "data/processed/train_features.csv",
    "extended": "data/processed/train_features_extended.csv",
}

MODEL_PATHS = {
    "baseline": "models/rf_baseline.joblib",
    "extended": "models/rf_extended.joblib",
}

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
    args = parser.parse_args()

    df = pl.read_csv(FEATURE_PATHS[args.features])
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    x = df.select(feature_cols).to_numpy()
    y = df["Survived"].to_numpy()

    x_train, x_val, y_train, y_val = train_test_split(x, y, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    model_path = MODEL_PATHS[args.features]
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()

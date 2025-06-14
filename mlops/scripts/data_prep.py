"""This script prepares processed Titanic datasets."""

import polars as pl

RAW_PATH = "data/raw/train.csv"
OUT_BASELINE = "data/processed/train_features.csv"
OUT_EXTENDED = "data/processed/train_features_extended.csv"


def prepare_baseline(df: pl.DataFrame) -> pl.DataFrame:
    """
    Prepare baseline features for the Titanic dataset.

    Fills missing values, encodes 'Sex', and applies one-hot encoding to 'Embarked'.

    Args:
        df (pl.DataFrame): Raw Titanic dataframe.

    Returns:
        pl.DataFrame: Dataframe with baseline features.
    """
    df = df.with_columns(
        [
            (pl.col("Sex") == "male").cast(pl.Int8).alias("Sex"),
            pl.col("Age").fill_null(pl.col("Age").median()),
            pl.col("Embarked").fill_null(pl.col("Embarked").mode()),
        ]
    )
    embarked_dummies = (
        df.select(pl.col("Embarked"))
        .to_dummies(drop_first=True)
        .with_columns([pl.all().cast(pl.Int8)])
    )
    df = df.hstack(embarked_dummies)
    return df


def prepare_extended(df: pl.DataFrame) -> pl.DataFrame:
    """
    Prepare extended features for the Titanic dataset.

    Adds family size, is-alone flag, and name length to the baseline features.

    Args:
        df (pl.DataFrame): Titanic dataframe after baseline processing.

    Returns:
        pl.DataFrame: Dataframe with extended features.
    """
    df = prepare_baseline(df)
    df = df.with_columns(
        [
            (pl.col("SibSp") + pl.col("Parch") + 1).alias("FamilySize"),
            ((pl.col("SibSp") + pl.col("Parch") + 1 == 1).cast(pl.Int8)).alias(
                "IsAlone"
            ),
            pl.col("Name").str.len_chars().alias("NameLength"),
        ]
    )
    return df


def main() -> None:
    """Main function to process Titanic data and save processed datasets."""
    df = pl.read_csv(RAW_PATH)

    df_baseline = prepare_baseline(df)
    df_baseline.write_csv(OUT_BASELINE)

    df_extended = prepare_extended(df)
    df_extended.write_csv(OUT_EXTENDED)

    print(f"Saved: {OUT_BASELINE}, {OUT_EXTENDED}")


if __name__ == "__main__":
    main()

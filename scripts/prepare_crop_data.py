from app.core.crop_model_config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN
)

from pathlib import Path

import pandas as pd

from app.core.crop_model_config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN
)


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "training" / "crop_recommendation.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "crop_model_data.csv"


def clean_text(series):
    return (
        series.astype("string")
        .str.replace("\u00a0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )


def prepare_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("Original dataset shape:", df.shape)

    # Validate required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Clean categorical features and target
    for col in CATEGORICAL_FEATURES + [TARGET_COLUMN]:
        df[col] = clean_text(df[col])

    # Convert numerical features
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Select only the required columns
    prepared = df[REQUIRED_COLUMNS].copy()

    # Check missing values after processing
    print("\nMissing values:")
    print(prepared.isnull().sum().to_string())

    if prepared.isnull().any().any():
        raise ValueError(
            "Missing values found. Check the original dataset."
        )

    # Check empty categorical values
    text_columns = CATEGORICAL_FEATURES + [TARGET_COLUMN]

    for col in text_columns:
        if prepared[col].str.strip().eq("").any():
            raise ValueError(
                f"Empty values found in column: {col}"
            )

    # Save prepared data
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    prepared.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n========== PREPARATION SUMMARY ==========")

    print("Prepared dataset shape:", prepared.shape)

    print("Input features:", len(FEATURE_COLUMNS))

    print("Target column:", TARGET_COLUMN)

    print("Unique crops:", prepared[TARGET_COLUMN].nunique())

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    prepare_data()

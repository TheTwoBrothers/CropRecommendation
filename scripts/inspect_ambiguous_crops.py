from pathlib import Path

import pandas as pd

from app.core.crop_model_config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)
ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data" / "processed" / "crop_model_data.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"

AMBIGUOUS_CROPS = [
    "ribbed gourd",
    "annual moringa",
    "vegetable cowpea",
    "tomato",
    "muskmelon",
    "tinda"
]
def join_values(series):
    values = sorted(series.dropna().unique())
    return " | ".join(values)

def inspect_crops():

    df = pd.read_csv(DATA_FILE)

    ambiguous_df = df[
        df[TARGET_COLUMN].isin(AMBIGUOUS_CROPS)
    ].copy()

    print("Original dataset:", df.shape)
    print("Selected dataset:", ambiguous_df.shape)

    print("\nCrop distribution:")
    print(
        ambiguous_df[TARGET_COLUMN]
        .value_counts()
        .to_string()
    )
    print("\n========== CATEGORICAL DISTRIBUTIONS ==========")

    for col in CATEGORICAL_FEATURES:

        print(f"\nDistribution of {col}:")

        distribution = pd.crosstab(
            ambiguous_df[TARGET_COLUMN],
            ambiguous_df[col],
            normalize="index"
        ) * 100

        print(distribution.round(2).to_string())
    print("\n========== NUMERICAL DISTRIBUTIONS ==========")

    numerical_summary = (
        ambiguous_df.groupby(TARGET_COLUMN)[NUMERIC_FEATURES]
        .agg(["min", "median", "max", "mean"])
        .round(2)
    )

    print(numerical_summary.to_string())  
    print("\n========== EXACT CONFLICTING COMBINATIONS ==========")

    combinations = ambiguous_df.groupby(
        FEATURE_COLUMNS,
        dropna=False
    ).agg(
        crop_count=(TARGET_COLUMN, "nunique"),
        crop_labels=(TARGET_COLUMN, join_values)
    ).reset_index()

    conflicts = combinations[
        combinations["crop_count"] > 1
    ].copy()

    print("Conflicting combinations:", len(conflicts))

    print(conflicts.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conflicts.to_csv(
        OUTPUT_DIR / "ambiguous_crop_combinations.csv",
        index=False
    )

    print("\nConflict report saved successfully.")          

if __name__ == "__main__":
    inspect_crops()
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

INPUT_FILE=ROOT/"data"/"training"/"crop_recommendation.csv"
OUTPUT_FILE=ROOT/"data"/"processed"/"crop_model_data.csv"


TEXT_COLUMNS=[
    "CROPS",
    "SOIL",
    "SEASON",
    "SOWN",
    "WATER_SOURCE"
]

FEATURE_COLUMNS = [
    "SOIL",
    "SEASON",
    "SOWN",
    "WATER_SOURCE",
    "TEMP",
    "MAX_TEMP",
    "RELATIVE_HUMIDITY",
    "RELATIVE_HUMIDITY_MAX",
]

TARGET_COLUMN = "CROPS"


def clean_text(series):
    return (
        series.astype("string")
        .str.replace("\u00a0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )

def prepare_data():
    if not INPUT_FILE:
        raise FileNotFoundError(f"Dataset not  found {INPUT_FILE}")
    df=pd.read_csv(INPUT_FILE)

    print("Original dataset shape",df.shape)

    required_columns=FEATURE_COLUMNS+[TARGET_COLUMN]

    missing_columns=[
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns :{missing_columns}")


    for col in TEXT_COLUMNS:
        df[col]=clean_text(df[col])

    numeric_columns=[
        "TEMP",
        "MAX_TEMP",
        "RELATIVE_HUMIDITY",
        "RELATIVE_HUMIDITY_MAX",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")   

    prepared = df[required_columns].copy()

    print("\nMissing values:")
    print(prepared.isnull().sum().to_string())

    if prepared.isnull().any().any():
        raise ValueError(
            "Missing values found. Investigate them before training."
        )
    
    if (prepared[TARGET_COLUMN] == "").any():
        raise ValueError("Empty crop labels found.")

    OUTPUT_FILE.parent.mkdir(parents=True,exist_ok=True)

    prepared.to_csv(OUTPUT_FILE,index=False)

    print("\nPrepared dataset shape:", prepared.shape)
    print("Number of input features:", len(FEATURE_COLUMNS))
    print("\nCrop distribution:")
    print(prepared[TARGET_COLUMN].value_counts().to_string())

    print("\nSoil labels after cleaning:")
    print(prepared["SOIL"].nunique())

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    prepare_data()

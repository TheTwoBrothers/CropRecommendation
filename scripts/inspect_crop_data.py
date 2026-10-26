from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

DATA_FILE=ROOT/"data"/"training"/"crop_recommendation.csv"


def inspect_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"dataset not found: {DATA_FILE}")

    df=pd.read_csv(DATA_FILE)

    print("\n========== DATASET OVERVIEW ==========")
    print("File:", DATA_FILE.name)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\n========== COLUMN NAMES ==========")
    print(df.columns.tolist())

    print("\n========== DATA TYPES ==========")
    print(df.dtypes.to_string())

    print("\n========== MISSING VALUES ==========")
    missing = df.isnull().sum()
    print(missing.to_string())

    print("\nTotal missing values:", df.isnull().sum().sum())

    print("\n========== DUPLICATE ROWS ==========")
    print("Duplicate rows:", df.duplicated().sum())

    print("\n========== CROP LABELS ==========")
    crop_counts = df["CROPS"].astype("string").str.strip().value_counts()

    print("Number of crop labels:", len(crop_counts))
    print("Minimum rows per crop:", crop_counts.min())
    print("Maximum rows per crop:", crop_counts.max())
    print(crop_counts.to_string())

    print("\n========== SOIL LABELS ==========")

    soil = df["SOIL"].astype("string")

    # Standardize whitespace, including non-breaking spaces
    soil_clean = (
        soil.str.replace("\u00a0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )

    print("Raw soil labels:", soil.nunique())
    print("After basic text normalization:", soil_clean.nunique())

    print("\nMost common normalized soil labels:")
    print(soil_clean.value_counts().head(20).to_string())

    print("\n========== SEASON DISTRIBUTION ==========")
    print(df["SEASON"].value_counts(dropna=False).to_string())

    print("\n========== IRRIGATION DISTRIBUTION ==========")
    print(df["WATER_SOURCE"].value_counts(dropna=False).to_string())

    numeric_columns = [
        "SOIL_PH",
        "CROPDURATION",
        "TEMP",
        "MAX_TEMP",
        "WATERREQUIRED",
        "WATERREQUIRED_MAX",
        "RELATIVE_HUMIDITY",
        "RELATIVE_HUMIDITY_MAX",
    ]

    print("\n========== NUMERIC FEATURE SUMMARY ==========")
    print(df[numeric_columns].describe().round(2).to_string())

    print("\n========== FIRST FIVE ROWS ==========")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    inspect_data() 
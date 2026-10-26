from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

DATA_FILE=ROOT/"data"/"training"/"crop_recommendation.csv"

def analyze_data():

    df=pd.read_csv(DATA_FILE)
    print("\n========== DATASET ==========")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))


    print("\n========== UNIQUE VALUES ==========")

    columns=[
        "CROPS",
        "SOIL",
        "SOWN",
        "WATER_SOURCE",
    ]

    for col in columns:
        print(f"{col}: {df[col].nunique()}")

    print("\n========== WATER REQUIREMENT ==========")    

    water_columns = [
        "WATERREQUIRED",
        "WATERREQUIRED_MAX",
    ]
    print(df[water_columns].describe().round(2))

    valid_range = (
        df["WATERREQUIRED"] <= df["WATERREQUIRED_MAX"]
    )

    print(
        "\nRows where WATERREQUIRED <= WATERREQUIRED_MAX:",
        valid_range.sum(),
        "out of",
        len(df),
    )

    # Summarize soil and season diversity for each crop
    print("\n========== CROP-WISE SUMMARY ==========")

    summary = df.groupby("CROPS").agg(
        rows=("CROPS", "size"),
        soil_types=("SOIL", "nunique"),
        seasons=("SEASON", "nunique"),
        irrigation_types=("WATER_SOURCE", "nunique"),
        temperature_min=("TEMP", "min"),
        temperature_max=("MAX_TEMP", "max"),
        water_required_min=("WATERREQUIRED", "min"),
        water_required_median=("WATERREQUIRED", "median"),
        water_required_observed_max=("WATERREQUIRED", "max"),
        water_required_max_column=("WATERREQUIRED_MAX", "max"),
    )

    print(summary.head(15).round(2).to_string())

    # Check whether the same input combination maps to multiple crops
    print("\n========== INPUT COMBINATION CHECK ==========")

    feature_columns = [
        "SOIL",
        "SEASON",
        "SOWN",
        "WATER_SOURCE",
        "TEMP",
        "MAX_TEMP",
        "RELATIVE_HUMIDITY",
        "RELATIVE_HUMIDITY_MAX",
    ]

    unique_inputs = df[feature_columns].drop_duplicates()

    print("Unique input combinations:", len(unique_inputs))
    print(
        "Total input rows:",
        len(df),
    )

    crop_counts = df.groupby(feature_columns, dropna=False)[
        "CROPS"
    ].nunique()

    conflicts = crop_counts[crop_counts > 1]

    print(
        "Identical input combinations associated with multiple crops:",
        len(conflicts),
    )


if __name__ == "__main__":
    analyze_data()
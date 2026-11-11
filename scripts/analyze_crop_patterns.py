from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "training" / "crop_recommendation.csv"
OUTPUT_DIR = ROOT / "data" / "processed"


def clean_text(series):
    return (
        series.astype("string")
        .str.replace("\u00a0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )


def join_values(series):
    values = sorted(series.dropna().unique())
    return " | ".join(values)


def analyze_patterns():

    df = pd.read_csv(DATA_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Normalize categorical columns
    text_columns = [
        "CROPS",
        "SOIL",
        "SEASON",
        "SOWN",
        "WATER_SOURCE"
    ]

    for col in text_columns:
        df[col] = clean_text(df[col])

    print("\n========== CROP-WISE PATTERNS ==========")

    # Create a profile for every crop
    profile = df.groupby("CROPS").agg(
        rows=("CROPS", "size"),

        soil_type_count=("SOIL", "nunique"),
        soil_types=("SOIL", join_values),

        season_count=("SEASON", "nunique"),
        seasons=("SEASON", join_values),

        sowing_month_count=("SOWN", "nunique"),
        sowing_months=("SOWN", join_values),

        irrigation_type_count=("WATER_SOURCE", "nunique"),
        irrigation_types=("WATER_SOURCE", join_values),

        temperature_min=("TEMP", "min"),
        temperature_mean=("TEMP", "mean"),
        temperature_max=("TEMP", "max"),

        water_required_min=("WATERREQUIRED", "min"),
        water_required_median=("WATERREQUIRED", "median"),
        water_required_max_observed=("WATERREQUIRED", "max"),

        water_required_max_column_min=("WATERREQUIRED_MAX", "min"),
        water_required_max_column_max=("WATERREQUIRED_MAX", "max")
    ).reset_index()

    print(profile.head(15).to_string(index=False))

    profile.to_csv(
        OUTPUT_DIR / "crop_profile.csv",
        index=False
    )

    # Check crops restricted to a single soil type
    print("\n========== SOIL TYPE RESTRICTIONS ==========")

    single_soil = profile[
        profile["soil_type_count"] == 1
    ]

    print("Crops having only one soil type:")
    print(single_soil[["CROPS", "soil_types"]].to_string(index=False))

    # Check seasonal restrictions
    print("\n========== SEASON RESTRICTIONS ==========")

    single_season = profile[
        profile["season_count"] == 1
    ]

    print("Crops having only one season:")
    print(
        single_season[["CROPS", "seasons"]].to_string(index=False)
    )

    # Crop versus season distribution
    season_table = pd.crosstab(
        df["CROPS"],
        df["SEASON"]
    )

    season_table.to_csv(
        OUTPUT_DIR / "crop_season_counts.csv"
    )

    # Crop versus irrigation distribution
    irrigation_table = pd.crosstab(
        df["CROPS"],
        df["WATER_SOURCE"]
    )

    irrigation_table.to_csv(
        OUTPUT_DIR / "crop_irrigation_counts.csv"
    )

    # Crop versus soil distribution
    soil_table = pd.crosstab(
        df["CROPS"],
        df["SOIL"]
    )

    soil_table.to_csv(
        OUTPUT_DIR / "crop_soil_counts.csv"
    )

    # Check identical inputs associated with different crops
    print("\n========== CONFLICTING INPUT COMBINATIONS ==========")

    features = [
        "SOIL",
        "SEASON",
        "SOWN",
        "WATER_SOURCE",
        "TEMP",
        "MAX_TEMP",
        "RELATIVE_HUMIDITY",
        "RELATIVE_HUMIDITY_MAX"
    ]

    conflicts = (
        df.groupby(features, dropna=False)["CROPS"]
        .nunique()
    )

    conflicts = conflicts[conflicts > 1]

    print("Conflicting combinations:", len(conflicts))

    print("\n========== REPORTS SAVED ==========")
    print("crop_profile.csv")
    print("crop_season_counts.csv")
    print("crop_irrigation_counts.csv")
    print("crop_soil_counts.csv")


if __name__ == "__main__":
    analyze_patterns()
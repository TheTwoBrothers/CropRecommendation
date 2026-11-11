from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "training" / "crop_recommendation.csv"

OUTPUT_DIR = ROOT / "data" / "processed"

FEATURE_COLUMNS = [
    "SOIL",
    "SEASON",
    "SOWN",
    "WATER_SOURCE",
    "TEMP",
    "MAX_TEMP",
    "RELATIVE_HUMIDITY",
    "RELATIVE_HUMIDITY_MAX"
]
df = pd.read_csv(DATA_FILE)

text_columns = [
    "CROPS",
    "SOIL",
    "SEASON",
    "SOWN",
    "WATER_SOURCE"
]

for col in text_columns:
    df[col] = (
        df[col].astype("string")
        .str.replace("\u00a0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

profile = df.groupby("CROPS").agg(
    rows=("CROPS", "size"),
    soil_types=("SOIL", "nunique"),
    seasons=("SEASON", "nunique"),
    irrigation_types=("WATER_SOURCE", "nunique")
).reset_index()

print("\n========== ALL CROP PROFILES ==========")

print(profile.to_string(index=False))

print("\n========== SUMMARY ==========")

print("Total crops:", len(profile))

print(
    "Crops with one soil type:",
    (profile["soil_types"] == 1).sum()
)

print(
    "Crops with one season:",
    (profile["seasons"] == 1).sum()
)

print(
    "Crops with one irrigation type:",
    (profile["irrigation_types"] == 1).sum()
)

print("\n========== CONFLICTING COMBINATIONS ==========")

unique_input_crop_counts = (
    df.groupby(FEATURE_COLUMNS, dropna=False)["CROPS"]
    .nunique()
)

conflict_keys = (
    unique_input_crop_counts[
        unique_input_crop_counts > 1
    ].reset_index()[FEATURE_COLUMNS]
)

conflict_rows = df.merge(
    conflict_keys,
    on=FEATURE_COLUMNS,
    how="inner"
)

conflict_report = (
    conflict_rows[FEATURE_COLUMNS + ["CROPS"]]
    .drop_duplicates()
    .sort_values(FEATURE_COLUMNS)
)

print("Conflicting combinations:", len(conflict_keys))

print(conflict_report.to_string(index=False))

conflict_report.to_csv(
    OUTPUT_DIR / "conflicting_input_combinations.csv",
    index=False
)
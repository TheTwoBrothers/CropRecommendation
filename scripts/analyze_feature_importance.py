from pathlib import Path

import joblib
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from app.core.crop_model_config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE
)


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "crop_model_data.csv"

MODEL_FILE = ROOT / "models" / "crop_recommender.joblib"

OUTPUT_FILE = ROOT / "data" / "processed" / "baseline_feature_importance.csv"


def analyze_feature_importance():

    # Load the prepared dataset
    df = pd.read_csv(DATA_FILE)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Recreate the same test dataset
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Load the trained pipeline
    pipeline = joblib.load(MODEL_FILE)

    print("Model loaded successfully.")

    # Calculate permutation importance
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="f1_macro",
        n_jobs=1
    )

    # Prepare the results
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    })

    # Sort by importance
    importance_df = importance_df.sort_values(
        "importance_mean",
        ascending=False
    )

    print("\n========== FEATURE IMPORTANCE ==========")
    print(importance_df.to_string(index=False))

    # Save the results
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    importance_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFeature importance report saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    analyze_feature_importance()
    
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from app.core.crop_model_config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    TOP_K
)
ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "crop_model_data.csv"

MODEL_FILE = ROOT / "models" / "crop_recommender.joblib"

OUTPUT_DIR = ROOT / "data" / "processed"

def analyze_errors():

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    pipeline = joblib.load(MODEL_FILE)

    print("Model loaded successfully.")
    print("Test records:", len(X_test))

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)

    classes = pipeline.named_steps["classifier"].classes_
    
    top_k = min(TOP_K, len(classes))

    top_indices = np.argsort(
        probabilities,
        axis=1
    )[:, ::-1][:, :top_k]

    top_candidates = classes[top_indices]

    errors = X_test.copy()

    errors["actual_crop"] = y_test

    errors["predicted_crop"] = pd.Series(
        predictions,
        index=X_test.index
    )

    errors["top_3_candidates"] = [
        " | ".join(row)
        for row in top_candidates
    ]

    errors["top_1_score"] = probabilities.max(axis=1)

    errors = errors[
        errors["actual_crop"] != errors["predicted_crop"]
    ].copy()

    wrong_pairs = (
        errors.groupby(
            ["actual_crop", "predicted_crop"]
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    wrong_by_crop = (
        errors.groupby("actual_crop")
        .size()
        .reset_index(name="wrong_predictions")
        .sort_values("wrong_predictions", ascending=False)
    )

    print("\n========== ERROR SUMMARY ==========")

    print("Total wrong predictions:", len(errors))

    print("\n========== MOST COMMON CONFUSIONS ==========")

    print(wrong_pairs.head(20).to_string(index=False))

    print("\n========== ERRORS BY ACTUAL CROP ==========")

    print(wrong_by_crop.to_string(index=False))

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    errors.to_csv(
        OUTPUT_DIR / "baseline_wrong_predictions.csv",
        index=False
    )

    wrong_pairs.to_csv(
        OUTPUT_DIR / "baseline_misclassified_pairs.csv",
        index=False
    )

    print("\nError reports saved successfully.")

if __name__ == "__main__":
    analyze_errors()
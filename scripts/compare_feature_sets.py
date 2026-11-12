from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.core.crop_model_config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    TOP_K
)
ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "crop_model_data.csv"

OUTPUT_FILE = ROOT / "data" / "processed" / "feature_comparison.csv"

def evaluate_feature_set(
    df,
    categorical_features,
    numeric_features,
    model_name,
    train_indices,
    test_indices
):
    feature_columns = categorical_features + numeric_features

    X_train = df.loc[train_indices, feature_columns]
    X_test = df.loc[test_indices, feature_columns]

    y_train = df.loc[train_indices, TARGET_COLUMN]
    y_test = df.loc[test_indices, TARGET_COLUMN]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numeric_features
            )
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    probabilities = pipeline.predict_proba(X_test)

    classes = pipeline.named_steps["classifier"].classes_

    top_k = min(TOP_K, len(classes))

    top_indices = np.argsort(
        probabilities,
        axis=1
    )[:, -top_k:][:, ::-1]

    top_predictions = classes[top_indices]

    top_k_accuracy = np.any(
        top_predictions == y_test.to_numpy()[:, None],
        axis=1
    ).mean()

    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "top_k_accuracy": top_k_accuracy
    }
def main():

    df = pd.read_csv(DATA_FILE)

    train_indices, test_indices = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN]
    )

    results = []

    # Model A: All features
    results.append(
        evaluate_feature_set(
            df,
            CATEGORICAL_FEATURES,
            NUMERIC_FEATURES,
            "All Features",
            train_indices,
            test_indices
        )
    )

    # Model B: Remove SOWN
    reduced_categorical = [
        col for col in CATEGORICAL_FEATURES
        if col != "SOWN"
    ]

    results.append(
        evaluate_feature_set(
            df,
            reduced_categorical,
            NUMERIC_FEATURES,
            "Without SOWN",
            train_indices,
            test_indices
        )
    )

    comparison = pd.DataFrame(results)

    print("\n========== FEATURE COMPARISON ==========")
    print(comparison.to_string(index=False))

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()


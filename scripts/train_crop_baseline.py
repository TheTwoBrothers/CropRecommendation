from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.core.crop_model_config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    TOP_K
)

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data" / "processed" / "crop_model_data.csv"
)

MODEL_DIR = ROOT / "models"

MODEL_FILE = MODEL_DIR / "crop_recommender.joblib"

REPORT_FILE = (
    ROOT / "data" / "processed" / "crop_baseline_report.txt"
)

def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Missing values found in dataset.")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    print("Dataset shape:", df.shape)
    print("Number of features:", X.shape[1])
    print("Number of crop classes:", y.nunique())

    return X, y

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    return X_train, X_test, y_train, y_test

def create_preprocessor():

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            ),
            (
                "numerical",
                "passthrough",
                NUMERIC_FEATURES
            )
        ]
    )

    return preprocessor

def create_model(preprocessor):

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

    return pipeline

def evaluate_model(pipeline, X_test, y_test):

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

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

    print("\n========== MODEL EVALUATION ==========")

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    return accuracy, macro_f1, weighted_f1, report

def calculate_top_k_accuracy(pipeline, X_test, y_test):

    probabilities = pipeline.predict_proba(X_test)

    classes = pipeline.named_steps["classifier"].classes_

    top_k = min(TOP_K, len(classes))

    top_indices = np.argsort(
        probabilities,
        axis=1
    )[:, -top_k:][:, ::-1]

    top_predictions = classes[top_indices]

    correct = np.any(
        top_predictions == y_test.to_numpy()[:, None],
        axis=1
    )

    top_k_accuracy = correct.mean() * 100

    print(
        f"\nTop-{top_k} Accuracy:",
        round(top_k_accuracy, 2),
        "%"
    )

    return top_k_accuracy

def save_model(pipeline):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        MODEL_FILE
    )

    print("\nModel saved successfully:")
    print(MODEL_FILE)

def main():

    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = create_preprocessor()

    pipeline = create_model(preprocessor)

    print("\nTraining Random Forest...")

    pipeline.fit(X_train, y_train)

    accuracy, macro_f1, weighted_f1, report = evaluate_model(
        pipeline,
        X_test,
        y_test
    )

    top_k_accuracy = calculate_top_k_accuracy(
        pipeline,
        X_test,
        y_test
    )

    save_model(pipeline)

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Macro F1: {macro_f1:.4f}\n")
        file.write(f"Weighted F1: {weighted_f1:.4f}\n")
        file.write(f"Top-{TOP_K} Accuracy: {top_k_accuracy:.4f}\n\n")
        file.write(report)

    print("\nEvaluation report saved:", REPORT_FILE)


if __name__ == "__main__":
    main()

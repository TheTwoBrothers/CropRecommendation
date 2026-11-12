from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.core.crop_model_config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FEATURE_COLUMNS,
    TOP_K
)

from app.utils.text_preprocessing import normalize_text_series


ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = ROOT / "models" / "crop_recommender.joblib"


class CropRecommendationService:

    def __init__(self):

        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Crop model not found: {MODEL_FILE}"
            )

        # Load the complete trained pipeline
        self.pipeline = joblib.load(MODEL_FILE)

        # Access the trained classifier
        self.classifier = self.pipeline.named_steps["classifier"]

        # Access the fitted categorical encoder
        self.encoder = (
            self.pipeline
            .named_steps["preprocessor"]
            .named_transformers_["categorical"]
        )

        # Store categories learned during training
        self.known_categories = {
            column: set(categories)
            for column, categories in zip(
                CATEGORICAL_FEATURES,
                self.encoder.categories_
            )
        }

    def prepare_input(self, input_features):

        # Check required features
        missing_features = [
            col for col in FEATURE_COLUMNS
            if col not in input_features
        ]

        if missing_features:
            raise ValueError(
                f"Missing input features: {missing_features}"
            )

        # Create a one-row DataFrame
        input_df = pd.DataFrame(
            [
                {
                    col: input_features[col]
                    for col in FEATURE_COLUMNS
                }
            ],
            columns=FEATURE_COLUMNS
        )

        # Normalize categorical inputs
        for col in CATEGORICAL_FEATURES:
            input_df[col] = normalize_text_series(input_df[col])

            if input_df[col].isna().any():
                raise ValueError(
                    f"Missing value in feature: {col}"
                )

            if input_df[col].eq("").any():
                raise ValueError(
                    f"Empty value in feature: {col}"
                )

        # Convert numerical inputs
        for col in NUMERIC_FEATURES:

            input_df[col] = pd.to_numeric(
                input_df[col],
                errors="coerce"
            )

            if input_df[col].isna().any():
                raise ValueError(
                    f"Invalid numerical value in: {col}"
                )

            if not np.isfinite(input_df[col].iloc[0]):
                raise ValueError(
                    f"Non-finite numerical value in: {col}"
                )

        return input_df

    def recommend(self, input_features, top_k=TOP_K):

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        # Prepare the farmer's input
        input_df = self.prepare_input(input_features)

        # Check for categories not seen during training
        warnings = []

        for col in CATEGORICAL_FEATURES:

            value = input_df[col].iloc[0]

            if value not in self.known_categories[col]:
                warnings.append(
                    f"Unseen category '{value}' in {col}. "
                    "Prediction may be unreliable."
                )

        # Get prediction scores for all crop classes
        scores = self.pipeline.predict_proba(input_df)[0]

        # Get the names of all trained crop classes
        classes = self.classifier.classes_

        # Determine how many recommendations to return
        top_k = min(top_k, len(classes))

        # Get indices of highest scoring crops
        top_indices = np.argsort(scores)[::-1][:top_k]

        recommendations = []

        for rank, index in enumerate(top_indices, start=1):

            recommendations.append(
                {
                    "rank": rank,
                    "crop": str(classes[index]),
                    "model_score": round(float(scores[index]), 4)
                }
            )

        return {
            "recommendations": recommendations,
            "warnings": warnings
        }
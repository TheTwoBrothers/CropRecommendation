"""
Central configuration for the crop recommendation model.

This file defines the features, target and training settings
used across data preparation, training and inference.
"""

# Categorical features
CATEGORICAL_FEATURES = [
    "SOIL",
    "SEASON",
    "SOWN",
    "WATER_SOURCE"
]

# Numerical features
NUMERIC_FEATURES = [
    "TEMP",
    "MAX_TEMP",
    "RELATIVE_HUMIDITY",
    "RELATIVE_HUMIDITY_MAX"
]

# Combine categorical and numerical features
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Target column
TARGET_COLUMN = "CROPS"

# All columns required in the prepared training dataset
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


# Model training configuration
RANDOM_STATE = 42

TEST_SIZE = 0.20

TOP_K = 3
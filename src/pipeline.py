"""
pipeline.py

Purpose
-------
This file prepares the datasets before they are used for machine learning.

Responsibilities:
• Load the training and test datasets
• Clean missing values
• Prepare features for model training
"""

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def load_data():
    """Load the training and test datasets from the data folder."""
    # Define the location of the dataset files.
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    # Stop early if a required file is missing.
    if not train_path.exists():
        raise FileNotFoundError("Could not find train.csv inside the data folder.")

    if not test_path.exists():
        raise FileNotFoundError("Could not find test.csv inside the data folder.")

    # Read the files into pandas DataFrames.
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def preprocess_data(train_df, test_df):
    """Clean missing values so the datasets are ready for training."""
    # Create copies to keep the original data unchanged.
    clean_train_df = train_df.copy()
    clean_test_df = test_df.copy()

    # Find all numerical columns because numbers are cleaned differently.
    numeric_columns = clean_train_df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        if column in clean_test_df.columns:
            # Replace missing numbers with a fixed placeholder value.
            clean_train_df[column] = clean_train_df[column].fillna(0)
            clean_test_df[column] = clean_test_df[column].fillna(0)

    # Find all categorical columns because text values need a different approach.
    categorical_columns = clean_train_df.select_dtypes(exclude=[np.number]).columns
    for column in categorical_columns:
        if column in clean_test_df.columns:
            # Replace missing text values with the most common category.
            most_common_value = clean_train_df[column].mode(dropna=True)
            if not most_common_value.empty:
                fill_value = most_common_value.iloc[0]
                clean_train_df[column] = clean_train_df[column].fillna(fill_value)
                clean_test_df[column] = clean_test_df[column].fillna(fill_value)

    return clean_train_df, clean_test_df


def prepare_features(train_df, test_df):
    """Separate the target column and prepare features for the model."""
    # Separate the target and the identifier because neither should be used
    # as an input feature. Id is only needed later to build submission.csv.
    X_train = train_df.drop(columns=["SalePrice", "Id"])
    y_train = train_df["SalePrice"]
    X_test = test_df.drop(columns=["Id"])

    # Convert text values into numbers because models need numeric inputs.
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # Make sure both datasets have the same columns before training.
    X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)

    return X_train, y_train, X_test

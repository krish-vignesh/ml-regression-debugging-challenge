"""
model.py

Purpose
-------
This file creates and trains the machine learning model.

Responsibilities
• Build the model
• Train the model

This file does NOT
• Load data
• Preprocess data
• Evaluate performance
• Make predictions
"""

from sklearn.ensemble import RandomForestRegressor


def build_model():
    # Create the machine learning model.
    model = RandomForestRegressor(random_state=42, n_estimators=5)

    # Return the model so it can be trained later.
    return model


def train_model(model, X_train, y_train):
    # Train the model using the prepared training data.
    trained_model = model.fit(X_train, y_train)

    # Return the trained model so it can be used later.
    return trained_model

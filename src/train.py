"""
train.py

Purpose
-------
This file runs the complete machine learning workflow.

Responsibilities

• Load data
• Preprocess data
• Prepare features
• Build model
• Train model
• Evaluate performance on a validation split
• Generate predictions
• Save outputs

This file does NOT

• Tune hyperparameters
• Create submission files
"""

from pathlib import Path

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from pipeline import (
    load_data,
    preprocess_data,
    prepare_features,
)

from model import (
    build_model,
    train_model,
)

from evaluation import (
    calculate_metrics,
    calculate_cross_validation,
    calculate_score,
    save_metrics,
    display_training_report,
)


def main():
    # Load the datasets.
    train_df, test_df = load_data()

    # Clean missing values.
    train_df, test_df = preprocess_data(
        train_df,
        test_df,
    )

    # Prepare the training features.
    X_train, y_train, X_test = prepare_features(
        train_df,
        test_df,
    )

    # Set aside a validation split purely to measure how well the model
    # generalizes to unseen data. The final model below is still trained
    # on the full training set, so this does not change the saved outputs.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Create the machine learning model.
    model = build_model()

    # Train the model.
    trained_model = train_model(
        model,
        X_train,
        y_train,
    )

    validation_predictions = trained_model.predict(X_val)
    metrics = calculate_metrics(y_val, validation_predictions)

    # 5-Fold Cross Validation gives a more stable estimate of
    # generalization than the single hold-out split above, since it
    # trains and evaluates on 5 different splits of the same data.
    # It is an additional signal, not a replacement for the hold-out split.
    cv_metrics = calculate_cross_validation(build_model(), X_train, y_train)
    metrics.update(cv_metrics)

    metrics["score"] = calculate_score(metrics["mae"])
    save_metrics(metrics)

    # Generate predictions on the test data.
    predictions = trained_model.predict(X_test)

    # Create the outputs folder if needed.
    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Save the trained model.
    model_path = outputs_dir / "model.pkl"
    joblib.dump(trained_model, model_path)

    # Save the predictions.
    predictions_path = outputs_dir / "predictions.csv"
    predictions_df = pd.DataFrame({"SalePrice": predictions})
    predictions_df.to_csv(predictions_path, index=False)

    # Print success messages.
    print("Machine learning pipeline completed successfully.")
    print("Model saved to outputs/model.pkl")
    print("Predictions saved to outputs/predictions.csv")

    # Show the training evaluation report.
    display_training_report(
        metrics,
        model_saved=model_path.exists(),
        predictions_generated=predictions_path.exists(),
    )


if __name__ == "__main__":
    main()

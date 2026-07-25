"""
evaluation.py

Purpose
-------
This file is the single, shared place where evaluation metrics and the
challenge score are calculated and displayed. Both train.py and
submit.py import from here so the metric formulas only ever live in
one place.

Responsibilities
• Calculate MAE, RMSE, and R²
• Calculate a simple 0-100 challenge score
• Save/load metrics so submit.py can reuse numbers computed during training
• Print beginner-friendly evaluation reports
• Write the final evaluation_report.md / evaluation_report.json files

This file does NOT
• Train a model
• Load or preprocess data
• Make predictions
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
REPORT_MD_PATH = OUTPUTS_DIR / "evaluation_report.md"
REPORT_JSON_PATH = OUTPUTS_DIR / "evaluation_report.json"

SEPARATOR = "=" * 50
SUBSECTION_SEPARATOR = "-" * 40

# Simple educational scoring formula.
#
# score = 100 * (1 - MAE / REFERENCE_MAE), clipped to the 0-100 range.
#
# REFERENCE_MAE is the error level treated as "0 points" - it anchors
# the scale, the same way a naive/no-skill baseline anchors R2. Cutting
# MAE in half always earns the same number of points, no matter where
# you start, so the formula stays linear and easy to reason about:
# a lower MAE always means a higher score.
#
# The original version of this formula (score = 100 - MAE / 1000) used
# a divisor so large that the untouched starter pipeline already scored
# ~93/100. That left almost no visible headroom for participants who
# did the work of fixing preprocessing, feature engineering, and model
# selection - a 20-30% real-world reduction in error barely moved the
# number. REFERENCE_MAE is calibrated instead so the untouched baseline
# lands around 65-75 ("working baseline"), and genuine improvements are
# clearly visible as the score climbs toward 90-100 ("outstanding"):
#
#   60-70  -> working baseline
#   70-85  -> good improvement
#   85-95  -> excellent ML engineering
#   95-100 -> outstanding solution
#
# It is only used for this challenge, not a real-world metric.
SCORE_MAX = 100
REFERENCE_MAE = 24000


def calculate_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, and R2 for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def calculate_cross_validation(model, X, y, cv=5):
    """Run 5-fold cross validation and return the mean/std MAE.

    Cross validation retrains the given (unfitted) model on 5 different
    train/test splits of the SAME already-preprocessed X and y, giving
    a more stable estimate of generalization than a single hold-out
    split. It reuses whatever features the caller already prepared -
    it does not load data or preprocess anything itself.
    """
    neg_mae_scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    mae_scores = -neg_mae_scores

    return {
        "cv_mae_mean": float(mae_scores.mean()),
        "cv_mae_std": float(mae_scores.std()),
    }


def calculate_score(mae):
    """Convert MAE into a simple 0-100 educational challenge score.

    Formula: score = 100 * (1 - MAE / REFERENCE_MAE), clipped to 0-100.
    """
    score = SCORE_MAX * (1 - (mae / REFERENCE_MAE))
    return max(0.0, min(SCORE_MAX, score))


def save_metrics(metrics):
    """Save metrics to outputs/metrics.json.

    Predictions on the real test set have no ground-truth SalePrice,
    so submit.py has no way to recalculate MAE/RMSE/R2 itself. Saving
    the validation metrics here lets submit.py reuse them without
    duplicating any metric calculation logic.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_metrics():
    """Load previously saved metrics, or None if train.py hasn't run yet."""
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def _score_for(metrics):
    """Return the score already stored in metrics, or calculate it if missing."""
    if metrics.get("score") is not None:
        return metrics["score"]
    return calculate_score(metrics["mae"])


def display_training_report(metrics, model_saved, predictions_generated):
    """Print the evaluation report shown right after training.

    Shows Hold-Out Validation (the primary, single train/test split) and
    5-Fold Cross Validation (an additional, more stable estimate) side
    by side so participants can compare them.
    """
    score = _score_for(metrics)

    print(SEPARATOR)
    print("HEXAWARE AI DEBUGGING CHALLENGE")
    print("Training Evaluation")
    print(SEPARATOR)
    print("Hold-Out Validation")
    print(f"MAE  : {metrics['mae']:.2f}")
    print(f"RMSE : {metrics['rmse']:.2f}")
    print(f"R2   : {metrics['r2']:.4f}")
    print(SUBSECTION_SEPARATOR)
    print("5-Fold Cross Validation")
    print(f"Mean MAE           : {metrics['cv_mae_mean']:.2f}")
    print(f"Standard Deviation : {metrics['cv_mae_std']:.2f}")
    print(SUBSECTION_SEPARATOR)
    print(f"Final Score            : {score:.2f} / 100")
    print(f"Model Saved            : {'Yes' if model_saved else 'No'}")
    print(f"Predictions Generated  : {'Yes' if predictions_generated else 'No'}")
    print(SEPARATOR)


def display_final_report(metrics, submission_format_ok, prediction_count, submission_saved):
    """Print the official evaluation report shown by submit.py.

    Only reads values already present in metrics (loaded from
    metrics.json) - it never recalculates MAE, RMSE, R2, or cross
    validation itself.
    """
    score = _score_for(metrics)

    print(SEPARATOR)
    print("HEXAWARE AI DEBUGGING CHALLENGE")
    print()
    print("Official Evaluation")
    print(SEPARATOR)
    print(f"Submission Format      : {'Valid' if submission_format_ok else 'Invalid'}")
    print(f"Prediction Count       : {prediction_count}")
    print(f"Validation MAE         : {metrics['mae']:.2f}")
    print(f"Validation RMSE        : {metrics['rmse']:.2f}")
    print(f"Validation R2          : {metrics['r2']:.4f}")
    print(f"CV Mean MAE            : {metrics['cv_mae_mean']:.2f}")
    print(f"CV Standard Deviation  : {metrics['cv_mae_std']:.2f}")
    print(f"Final Score            : {score:.2f} / 100")
    print(f"submission.csv created : {'Yes' if submission_saved else 'No'}")
    print(SEPARATOR)


def _checklist_lines(items):
    """Turn a {label: True/False} dict into '- [check mark] label' lines."""
    lines = []
    for label, is_done in items.items():
        mark = "✅" if is_done else "❌"
        lines.append(f"- {mark} {label}")
    return lines


def _build_markdown_report(report_data, generated_files, repository_status):
    """Build the evaluation_report.md content as a list of lines.

    Only formats values that are already sitting in report_data - it
    does not calculate anything new.
    """
    lines = [
        "# Hexaware AI & Machine Learning Debugging Challenge",
        "",
        "## Final Evaluation Report",
        "",
        "---",
        "",
        "### Submission Status",
        "",
        f"**{report_data['submission_status']}**",
        "",
        f"**Submission File:** `{report_data['submission_file']}`",
        "",
        "---",
        "",
        "### Model Performance",
        "",
    ]

    if report_data["mae"] is None:
        lines.append("Training metrics unavailable. Run `python src/train.py` before submitting.")
    else:
        lines.append("**Hold-Out Validation**")
        lines.append("")
        lines.append(f"- MAE: {report_data['mae']:.2f}")
        lines.append(f"- RMSE: {report_data['rmse']:.2f}")
        lines.append(f"- R2: {report_data['r2']:.4f}")
        lines.append("")
        lines.append("----------------------------------------")
        lines.append("")
        lines.append("**5-Fold Cross Validation**")
        lines.append("")
        lines.append(f"- Mean MAE: {report_data['cv_mae_mean']:.2f}")
        lines.append(f"- Standard Deviation: {report_data['cv_mae_std']:.2f}")
        lines.append("")
        lines.append("----------------------------------------")
        lines.append("")
        lines.append(f"**Final Score:** {report_data['score']:.2f} / 100")

    lines += [
        "",
        "---",
        "",
        "### Generated Files",
        "",
        *_checklist_lines(generated_files),
        "",
        "---",
        "",
        "### Repository Status",
        "",
        *_checklist_lines(repository_status),
        "",
        "---",
        "",
        "### Timestamp",
        "",
        report_data["timestamp"],
        "",
    ]
    return lines


def generate_evaluation_reports(metrics, submission_status, prediction_count, submission_file, generated_files, repository_status):
    """Write outputs/evaluation_report.md and outputs/evaluation_report.json.

    metrics may be None if train.py has not produced metrics.json yet -
    the reports are still generated, just noting that metrics are
    unavailable. This function only formats and saves numbers that
    already exist; it never calculates MAE/RMSE/R2 itself.
    """
    score = _score_for(metrics) if metrics else None

    report_data = {
        "submission_status": submission_status,
        "mae": metrics["mae"] if metrics else None,
        "rmse": metrics["rmse"] if metrics else None,
        "r2": metrics["r2"] if metrics else None,
        "cv_mae_mean": metrics["cv_mae_mean"] if metrics else None,
        "cv_mae_std": metrics["cv_mae_std"] if metrics else None,
        "score": score,
        "prediction_count": prediction_count,
        "submission_file": submission_file,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_JSON_PATH.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    markdown_lines = _build_markdown_report(report_data, generated_files, repository_status)
    REPORT_MD_PATH.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    return REPORT_MD_PATH, REPORT_JSON_PATH

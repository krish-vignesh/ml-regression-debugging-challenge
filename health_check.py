from __future__ import annotations

import sys
import time
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - defensive path
    pd = None


REPO_ROOT = Path(__file__).resolve().parent
HEALTH_CHECK_VERSION = "1.0.0"
OUTPUTS_DIR = REPO_ROOT / "outputs"
REPORT_PATH = OUTPUTS_DIR / "health_report.txt"

SUCCESS_ICON = "✅"
FAILURE_ICON = "❌"
WARNING_ICON = "⚠️"
INFO_ICON = "ℹ️"
SECTION_SEPARATOR = "=" * 50
SUBSECTION_SEPARATOR = "-" * 40

REQUIRED_FILES = [
    "README.md",
    "Developer_Handover.md",
    "requirements.txt",
    "submit.py",
    "src",
    "src/pipeline.py",
    "src/model.py",
    "src/train.py",
    "data",
    "data/train.csv",
    "data/test.csv",
    "data/sample_submission.csv",
    "outputs",
]

REQUIRED_DEPENDENCIES = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("joblib", "joblib"),
]

MIN_PYTHON_VERSION = (3, 10)
WEIGHTS = {
    "python": 20,
    "dependencies": 20,
    "repository": 30,
    "dataset_loaded": 15,
    "dataset_health": 15,
}


def check_python_version() -> tuple[str, bool]:
    """Return the current Python version and whether it meets the minimum requirement."""
    try:
        version_text = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        is_supported = sys.version_info >= MIN_PYTHON_VERSION
    except Exception:  # pragma: no cover - defensive path
        version_text = "unknown"
        is_supported = False
    return version_text, is_supported


def check_dependencies() -> dict[str, bool]:
    """Check whether the required Python dependencies are available."""
    availability: dict[str, bool] = {}
    for display_name, module_name in REQUIRED_DEPENDENCIES:
        try:
            import_module(module_name)
            availability[display_name] = True
        except Exception:
            availability[display_name] = False
    return availability


def check_repository_structure() -> dict[str, bool]:
    """Verify that the expected repository files and directories exist."""
    structure_status: dict[str, bool] = {}
    for path_str in REQUIRED_FILES:
        path = REPO_ROOT / path_str
        structure_status[path_str] = path.exists()
    return structure_status


def check_dataset() -> tuple[bool, Any | None, str]:
    """Try to load the training dataset and return its status and contents."""
    if pd is None:
        return False, None, "pandas is not installed, so the dataset could not be loaded."

    train_path = REPO_ROOT / "data" / "train.csv"
    try:
        if not train_path.exists():
            return False, None, "The training dataset file could not be found."

        df = pd.read_csv(train_path)
        return True, df, ""
    except FileNotFoundError:
        return False, None, "The training dataset file could not be found."
    except pd.errors.EmptyDataError:
        return False, None, "The training dataset file is empty."
    except pd.errors.ParserError:
        return False, None, "The training dataset file could not be parsed correctly."
    except Exception as exc:  # pragma: no cover - defensive path
        return False, None, f"The training dataset could not be opened: {exc}"


def check_dataset_health(df: Any | None) -> dict[str, Any]:
    """Collect basic dataset health indicators without performing ML-specific analysis."""
    if df is None:
        return {
            "total_missing_values": None,
            "duplicate_rows": None,
            "column_count": None,
            "row_count": None,
        }

    return {
        "total_missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_count": int(df.shape[1]),
        "row_count": int(df.shape[0]),
    }


def _calculate_repository_score(
    python_supported: bool,
    dependency_status: dict[str, bool],
    structure_status: dict[str, bool],
    dataset_loaded: bool,
    dataset_health: dict[str, Any],
) -> int:
    """Calculate a simple readiness score for the repository."""
    score = 0

    if python_supported:
        score += WEIGHTS["python"]

    if dependency_status:
        dependency_ratio = sum(dependency_status.values()) / len(dependency_status)
        score += int(dependency_ratio * WEIGHTS["dependencies"])

    if structure_status:
        structure_ratio = sum(structure_status.values()) / len(structure_status)
        score += int(structure_ratio * WEIGHTS["repository"])

    if dataset_loaded:
        score += WEIGHTS["dataset_loaded"]

    if dataset_loaded and dataset_health.get("row_count") is not None:
        score += WEIGHTS["dataset_health"]

    return min(score, 100)


def print_summary(
    python_version: str,
    python_supported: bool,
    dependency_status: dict[str, bool],
    structure_status: dict[str, bool],
    dataset_loaded: bool,
    dataset_df: Any | None,
    dataset_error: str,
    dataset_health: dict[str, Any],
) -> None:
    """Print a professional repository health report and save it to disk."""
    score = _calculate_repository_score(
        python_supported=python_supported,
        dependency_status=dependency_status,
        structure_status=structure_status,
        dataset_loaded=dataset_loaded,
        dataset_health=dataset_health,
    )

    found_count = sum(1 for exists in structure_status.values() if exists)
    missing_count = len(structure_status) - found_count

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append(SECTION_SEPARATOR)
    lines.append("🏥 Hexaware AI & ML Debugging Challenge")
    lines.append("Repository Health Report")
    lines.append(SECTION_SEPARATOR)
    lines.append("")
    lines.append(f"{INFO_ICON} Repository Health Checker")
    lines.append(f"Version {HEALTH_CHECK_VERSION}")
    lines.append(f"Generated At {timestamp}")
    lines.append("")
    lines.append(f"{INFO_ICON} Python Environment")
    lines.append(SUBSECTION_SEPARATOR)
    lines.append(f"Version : {python_version}")
    lines.append(f"Status : {'Supported' if python_supported else 'Python 3.10+ recommended'}")
    lines.append("")
    lines.append(f"{INFO_ICON} Dependencies")
    lines.append(SUBSECTION_SEPARATOR)
    missing_dependencies: list[str] = []
    for dependency, available in dependency_status.items():
        if available:
            lines.append(f"{SUCCESS_ICON} {dependency}")
        else:
            missing_dependencies.append(dependency)
            lines.append(f"{FAILURE_ICON} Missing Dependency: {dependency}")

    if missing_dependencies:
        lines.append("")
        lines.append("Install dependencies using")
        lines.append("pip install -r requirements.txt")
    lines.append("")
    lines.append(f"{INFO_ICON} Repository Structure")
    lines.append(SUBSECTION_SEPARATOR)
    lines.append(f"Files Found : {found_count}")
    lines.append(f"Files Missing : {missing_count}")
    lines.append(f"Total Required : {len(structure_status)}")
    lines.append(f"Total Found : {found_count}")
    lines.append(f"Total Missing : {missing_count}")
    for path_str, exists in structure_status.items():
        status_icon = SUCCESS_ICON if exists else FAILURE_ICON
        lines.append(f"{status_icon} {path_str}")
    lines.append("")
    lines.append(f"{INFO_ICON} Summary")
    lines.append(SUBSECTION_SEPARATOR)
    lines.append(f"Found : {found_count}")
    lines.append(f"Missing : {missing_count}")
    lines.append("")
    lines.append(f"{INFO_ICON} Dataset")
    lines.append(SUBSECTION_SEPARATOR)
    if dataset_loaded and dataset_df is not None:
        lines.append("Status : Dataset loaded successfully")
        lines.append(f"Rows : {dataset_health['row_count']}")
        lines.append(f"Columns : {dataset_health['column_count']}")
        lines.append(f"Missing Values : {dataset_health['total_missing_values']}")
        lines.append(f"Duplicate Rows : {dataset_health['duplicate_rows']}")
    else:
        lines.append("Status : Unable to load the training dataset")
        lines.append(f"Details : {dataset_error or 'Please check that the file exists and is readable.'}")
    lines.append("")
    lines.append(SECTION_SEPARATOR)
    lines.append(f"Repository Score : {score} / 100")
    lines.append("")
    lines.append("Overall Status")
    if score >= 80:
        lines.append("Repository is ready for development.")
        lines.append("")
        lines.append("Next Step")
        lines.append("Run")
        lines.append("python src/train.py")
    else:
        lines.append("Repository requires attention before continuing.")
        lines.append("")
        lines.append("Next Step")
        lines.append("Please resolve the issues above before continuing.")
    lines.append("")
    lines.append(SECTION_SEPARATOR)
    lines.append("Health Check Completed Successfully")
    lines.append("Hexaware AI & ML Debugging Challenge")
    lines.append(f"Version {HEALTH_CHECK_VERSION}")
    lines.append(SECTION_SEPARATOR)

    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        lines.append("")
        lines.append(f"{WARNING_ICON} Warning")
        lines.append("Unable to save the report.")

    for line in lines:
        print(line)


def main() -> None:
    """Run all health checks and print a summary without crashing."""
    start_time = time.perf_counter()

    try:
        python_version, python_supported = check_python_version()
    except Exception:  # pragma: no cover - defensive path
        python_version = "unknown"
        python_supported = False

    try:
        dependency_status = check_dependencies()
    except Exception:  # pragma: no cover - defensive path
        dependency_status = {display_name: False for display_name, _ in REQUIRED_DEPENDENCIES}

    try:
        structure_status = check_repository_structure()
    except Exception:  # pragma: no cover - defensive path
        structure_status = {path_str: False for path_str in REQUIRED_FILES}

    try:
        dataset_loaded, dataset_df, dataset_error = check_dataset()
        dataset_health = check_dataset_health(dataset_df)
    except Exception:  # pragma: no cover - defensive path
        dataset_loaded = False
        dataset_df = None
        dataset_error = "The health check could not complete the dataset step."
        dataset_health = {
            "total_missing_values": None,
            "duplicate_rows": None,
            "column_count": None,
            "row_count": None,
        }

    print_summary(
        python_version=python_version,
        python_supported=python_supported,
        dependency_status=dependency_status,
        structure_status=structure_status,
        dataset_loaded=dataset_loaded,
        dataset_df=dataset_df,
        dataset_error=dataset_error,
        dataset_health=dataset_health,
    )

    elapsed_time = time.perf_counter() - start_time
    print()
    print("Health Check Completed")
    print(f"Time Taken : {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()

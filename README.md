# 🚀 AI & Machine Learning Debugging Challenge

**Estimated Time:** 60–90 Minutes
**Difficulty:** Beginner to Intermediate

---

# 📖 Overview

Welcome to the **AI & Machine Learning Debugging Challenge**.

You are given a working Machine Learning regression project. The code runs
successfully end-to-end: the model trains, predictions are generated, and a
submission file is produced.

However, the repository intentionally contains several Machine Learning
mistakes that quietly reduce model quality without causing any errors or
crashes. Nothing here is broken in an obvious "the code doesn't run" sense —
these are the kinds of issues that slip past a code review but show up as
weaker predictions, misleading metrics, or unnecessary noise in the data.

Your objective is to **identify, understand, and fix these issues** — and in
doing so, improve the model's real-world performance.

---

# 🎯 Objective

Participants should improve the model's predictive performance while
maintaining a technically correct Machine Learning pipeline.

"Technically correct" matters as much as "better score" — a fix that improves
a metric by cutting a methodological corner (for example, leaking information
from validation/test data into training) is not a real fix.

---

# 🛠️ What Participants May Change

You are free to modify the project. This is not a "find the one bug" puzzle —
treat it like a real codebase you've inherited and are being asked to improve.

Examples of fair game changes include:

- Fixing preprocessing and data cleaning logic
- Improving feature engineering
- Feature selection / removing noisy or low-value columns
- Hyperparameter tuning
- Trying different regression models, such as:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - CatBoost
  - LightGBM
  - Any other reasonable regression model

You're encouraged to experiment. **There is no single correct solution** —
different combinations of fixes and improvements can all be valid, well-reasoned
submissions.

---

# 📋 Rules

The repository must still, after your changes:

- ✅ Train successfully
- ✅ Generate predictions
- ✅ Produce `submission.csv`
- ✅ Preserve the overall project structure

A few things are off-limits regardless of how tempting they are for a better score:

- ❌ Do not intentionally hard-code predictions.
- ❌ Do not use the test labels anywhere in training or feature engineering.

---

# 🏆 Evaluation

Participants are evaluated using:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (coefficient of determination)
- **Cross Validation** performance

Higher model quality is better. A submission that improves these metrics
through sound Machine Learning practice will score higher than one that
achieves a better-looking number through an unsound shortcut.

---

# 💡 Tips

This challenge focuses on **Machine Learning reasoning**, not Python syntax.
You don't need to write clever code — you need to reason carefully about what
the pipeline is doing and whether it's doing it correctly.

To get the most out of your time:

- Read the preprocessing pipeline carefully — how is missing data handled?
  Does it make sense for every column?
- Inspect feature engineering opportunities — are there useful signals in the
  raw columns that aren't being surfaced for the model?
- Tune your models — don't assume the current settings are reasonable defaults.
- Compare validation metrics against each other — do they agree with one
  another, or is something suspicious going on?
- Think critically about any evaluation result that looks *too* good. In Machine
  Learning, a suspiciously great number is often a sign something is wrong
  rather than a sign of success.

---

# 🎁 Bonus Challenge

Once you've addressed the core issues, you're encouraged to go further than
the minimum needed to fix things. Ideas to explore:

- Trying gradient-boosted models like **XGBoost** or **CatBoost**
- Additional feature engineering beyond the obvious gaps
- More rigorous feature selection
- More thorough hyperparameter tuning (grid search, random search, etc.)

There is no single best solution — we're interested in seeing how you reason
about trade-offs and validate your improvements.

---

# 📁 Repository Structure

```text
.
├── README.md
├── requirements.txt
├── health_check.py
├── submit.py
├── .gitignore
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── datasets/
│   ├── original/
│   └── curated/
│
├── src/
│   ├── pipeline.py
│   ├── model.py
│   ├── train.py
│   └── evaluation.py
│
└── outputs/
```

---

# 🗂️ Dataset Organization

- **data/** – Contains the datasets used directly by the challenge.
- **datasets/original** – Contains the raw Ames Housing dataset.
- **datasets/curated** – Contains a cleaned reference dataset.

The pipeline always reads from **data/**. The **datasets/** folder exists
only for educational reference.

---

# 📖 Repository Guide

## 🩺 health_check.py

**Recommended starting point.**

Runs diagnostics on the repository and provides a summary of the current
project health.

Run this first before making any modifications.

---

## 📤 submit.py

Run this after completing your improvements.

This script validates your work and generates the required submission files
inside the **outputs/** directory.

---

## 📂 data/

Contains the datasets used in this challenge.

- **train.csv** – Training dataset
- **test.csv** – Dataset for prediction
- **sample_submission.csv** – Reference submission format

---

## 📂 datasets/

Provided for learning and reference purposes only.

- **original/** – Contains the original Ames Housing dataset.
- **curated/** – Contains a cleaned reference dataset used while preparing
  the challenge.

The Machine Learning pipeline uses the files inside **data/**, not this
folder.

---

## 📂 src/

Contains the complete Machine Learning implementation.

### pipeline.py

Responsible for:

- Loading data
- Data cleaning
- Missing value handling
- Feature engineering
- Encoding categorical variables
- Preparing data for training

---

### model.py

Responsible for:

- Creating the Machine Learning model
- Training
- Prediction
- Saving the trained model

---

### train.py

Main entry point of the project.

Executes the complete Machine Learning workflow:

1. Load Data
2. Preprocess Data
3. Train Model
4. Evaluate Model
5. Save Outputs

If you're unsure where to begin exploring the code, start here after
reviewing the health report.

---

### evaluation.py

Responsible for:

- Regression evaluation
- Metric calculation
- Evaluation report generation

---

## 📂 outputs/

Automatically generated after running the project.

Contains the generated outputs required for submission.

---

# 🧰 Requirements

- Python 3.10 or newer

---

# ⚙️ Environment Setup (Recommended)

Before installing the project dependencies, create a Python Virtual Environment.

### Windows

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

---

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Once activated, proceed with the installation steps below.

---

# ▶️ Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

# 🧭 Before You Start

This challenge intentionally contains Machine Learning engineering issues
rather than programming errors.

The repository runs successfully from end to end.

Your objective is to improve the Machine Learning pipeline using sound
engineering practices.

Focus on:

- preprocessing
- feature engineering
- feature selection
- validation methodology
- model selection
- hyperparameter tuning

Take time to understand the pipeline before modifying it.

---

# ▶️ Running the Project

### Step 1 — Check Repository Health

```bash
python health_check.py
```

---

### Step 2 — Train the Model

```bash
python src/train.py
```

---

### Step 3 — Generate Submission

```bash
python submit.py
```

---

# 🔍 Recommended Investigation Order

To make the best use of your time, we recommend following this sequence:

1. Run **health_check.py**
2. Explore **src/train.py**
3. Follow the execution flow into **pipeline.py** and **model.py**
4. Identify and fix Machine Learning issues
5. Retrain the model
6. Generate submission using **submit.py**

---

# 📦 Deliverables

Submit the files generated inside the **outputs/** directory after completing
the challenge.

---

# 🎉 Good Luck!

Approach this challenge as if you've joined a team and inherited an existing
Machine Learning project.

Your goal is not only to improve the model but also to demonstrate your
engineering thinking, debugging process, and ability to work with an
unfamiliar codebase.

**Happy Debugging! 🚀**

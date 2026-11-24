# Traffic Accident Severity Analysis

A beginner-friendly Python project that explores real US traffic accident records
and trains models to predict the severity category recorded in the dataset. It
includes **TRAFFIC INSIGHT**, a responsive analytics website built with Streamlit,
interactive Plotly analysis, model evaluation, and tests.

> **Important:** This project is for learning only. Its predictions are experimental,
> historical associations are not proof of cause, and it must not be used for
> emergency, legal, insurance, or real-world safety decisions.

## Problem and objectives

The project asks what patterns appear in the recorded severity classes across
weather, road conditions, lighting, time, speed limit (if present), temperature,
and visibility. It compares three models on the same held-out test split. It does
not claim those patterns cause accidents.

## Dataset

Use the real [Kaggle US Accidents dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).
It is a large dataset and Kaggle may require signing in. The current workspace
contains the real `data/accidents.csv` source file and the reproducible
100,000-row sample used for the saved analysis and model results.

1. Download the dataset from Kaggle and extract its archive.
2. Copy the CSV to `data/accidents.csv` (or use its common original filename
   `US_Accidents_March23.csv`).
3. See [data/README.md](data/README.md) for manual download and authenticated Kaggle
   CLI instructions.

The loader checks the real headers and requires the `Severity` target. It samples
up to 100,000 rows uniformly by reading the CSV in chunks. Set `max_rows=None` in
`load_dataset()` only if you have enough memory to load the full CSV. Optional
fields such as speed limit, lighting, or temperature are analyzed only when they
actually exist.

The current source contained 7,728,394 rows. Analysis and model results were
calculated using a 100,000-row sample (80,000 training rows and 20,000 held-out
test rows). The source data is not fabricated; do not commit the large CSV.

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly, scikit-learn, Streamlit, Joblib, and Pytest.

## Project layout

```text
Traffic-Accident-Severity-Analysis/
├── app.py                    # TRAFFIC INSIGHT multi-page dashboard
├── requirements.txt
├── README.md
├── data/
│   └── README.md
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train_model.py
│   └── evaluate_model.py
├── models/                  # Trained pipeline (generated locally)
├── outputs/
│   ├── plots/                # EDA and model charts
│   └── reports/              # EDA and evaluation results
└── tests/
```

## Install on Windows

Open a PowerShell terminal in the project directory:

```powershell
py -3.13 -m venv .venv313
.\.venv313\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, choose the `.venv313` interpreter using **Python: Select Interpreter**.
If PowerShell prevents environment activation, either enable scripts according to
your organization's policy or invoke `.venv313\Scripts\python.exe` directly in
the commands below.

## Run the workflow

After placing the CSV in `data/`:

```powershell
# Inspect and load a reproducible sample; fails with download guidance if absent
.\.venv313\Scripts\python.exe -m src.data_loader

# Check features and the leakage-safe train/test split
.\.venv313\Scripts\python.exe -m src.preprocessing

# Generate charts and a report calculated from your CSV
.\.venv313\Scripts\python.exe -m src.eda

# Train and compare all three models on the same 80/20 split
.\.venv313\Scripts\python.exe -m src.train_model

# Run tests
.\.venv313\Scripts\python.exe -m pytest tests -v

# Start the dashboard
.\.venv313\Scripts\python.exe -m streamlit run app.py
```

The website opens at `http://localhost:8501`. It loads a deterministic sample
of up to 100,000 dataset rows and presents six workspaces through horizontal
navigation: Home, Explore Data, Analytics, Severity Prediction, Model Performance,
and Reports. The app reuses the saved model and report files; it does not retrain
when pages change.

The latest saved evaluation selected **Random Forest** by macro F1: accuracy
0.6717, macro precision 0.4448, macro recall 0.4569, and macro F1 0.4464 on the
20,000 held-out records. Decision Tree macro F1 was 0.3562; Logistic Regression
macro F1 was 0.3136. The lowest-frequency class is very rare, so the scores are
moderate and class-specific performance should be reviewed in the saved report.

## Preprocessing and modeling

- The target is `Severity`; a missing target column is an explicit error.
- Exact duplicate rows and records with missing target labels are removed.
- `Start_Time` (or an available date field) is used to derive hour, weekday,
  month, and weekend features.
- IDs, post-accident end timestamps/coordinates, descriptions, casualty count,
  and high-cardinality location fields are excluded to limit leakage and
  overfitting. Other usable numeric and low-cardinality categorical features are
  selected from the real CSV columns.
- Numeric values are median-imputed and scaled; categories are imputed and
  one-hot encoded. These transformations are fitted only on each model's
  training partition via scikit-learn pipelines.
- Models: Logistic Regression, Decision Tree, and Random Forest. Class weights
  are used to reduce the impact of class imbalance.
- The default split is 80% training and 20% testing, stratified when every
  severity class has enough examples.
- Selection prioritizes macro F1, then macro recall, then accuracy, so rare
  severity classes are not ignored in favor of accuracy alone.

## Generated files

- `outputs/plots/`: severity distribution; road, weather, and lighting charts;
  hour, weekday, and month charts; numeric correlation; temperature/visibility
  when available; speed-limit chart only if a speed-limit field is present;
  confusion matrix and Random Forest importance when supported.
- `outputs/reports/eda_report.txt`: record counts, class balance, and available
  fields/statistics computed from the loaded CSV.
- `outputs/reports/model_comparison.csv`: accuracy, macro precision, macro recall,
  and macro F1 for all models on the same test split.
- `outputs/reports/evaluation_report.txt`: selected model and its classification
  report.
- `outputs/reports/confusion_matrix.csv` and
  `outputs/reports/feature_importance.csv`.
- `models/best_model.joblib`: selected full preprocessing-plus-classifier pipeline.

## Evaluation and limitations

Accuracy is the share of all correct predictions. Macro precision, recall, and
F1 calculate a score per class and average those scores equally. The confusion
matrix shows which classes are confused. These metrics describe one random
held-out sample; they do not establish future performance. Temporal/geographic
validation, class calibration, changing data distributions, missing variables,
and the dataset's sampling/recording process are important limitations.

## Explaining the project for a college review

In simple terms: first the program reads real accident rows, checks data quality,
and creates time-of-day features. Then it draws charts to describe the sample.
It separates records into training and testing groups; the models learn from
training data only, and the test group is used to compare predictions. The
dashboard lets a user inspect the data and try an experimental prediction. A
high model score is not evidence that a feature causes an accident or that the
model is safe for real decisions.

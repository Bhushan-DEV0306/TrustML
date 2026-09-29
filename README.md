# TrustML — Explainable AutoML Platform

Upload a CSV, pick what you want to predict, and TrustML analyzes your
data, warns you about problems (leakage, imbalance, missing values),
trains and compares multiple models, tunes the best ones, evaluates
honestly on held-out data, and gives you a ready-to-use model plus
prediction script — all with visible reasoning at every step.

**Live demo:** [add your Streamlit Cloud link here]

## Why this is different

Most beginner AutoML projects are "upload CSV → train Random Forest →
show accuracy." TrustML focuses on trust instead:

- **Explainable** — every cleaning step, encoding choice, and model
  decision is shown, not hidden.
- **Leakage-aware** — checks for suspicious features (near-perfect
  predictors, ID-like columns, suspicious names, date columns) before
  training, not after.
- **Metric-aware** — lets you choose the metric that fits your data
  (F1, recall, MAE, R², etc.), and Target tab suggests one based on
  class imbalance or target skew.
- **Honest evaluation** — every model is compared against a dummy
  baseline; accuracy is never claimed without cross-validation and a
  held-out test set.
- **Smart tuning** — only the top models are tuned with
  RandomizedSearchCV, not every model blindly.
- **Ready to use** — download the trained model and a standalone
  `predict.py` script to use it in your own project.

## Features

- CSV upload with full dataset analysis (rows, columns, dtypes,
  missing values, duplicates, cardinality)
- Automatic classification/regression detection with manual override
- Data-quality and leakage warnings: class imbalance, target
  skew/outliers, high correlation with target, a single feature that
  predicts the target too well, ID-like/free-text columns, suspicious
  column names, date columns
- Automatic cleaning: duplicates, constant columns, missing-heavy
  rows/columns, ID-like columns, missing target rows
- Smart numeric parsing for messy text columns (e.g. "2 BHK" → 2,
  "2100-2850" → 2475)
- Preprocessing pipeline: median/most-frequent imputation, scaling,
  one-hot encoding with rare-category grouping — built with
  scikit-learn's `ColumnTransformer` and `Pipeline`, fit only on
  training data
- Baseline model comparison via 5-fold cross-validation: Dummy,
  Logistic/Linear Regression, Decision Tree, Random Forest, Gradient
  Boosting, XGBoost, KNN
- Optional Lasso-based feature selection, kept only if it genuinely
  improves the score
- Hyperparameter tuning of the top models with RandomizedSearchCV,
  compared against untuned scores
- Final evaluation on a held-out test set: accuracy/precision/recall/F1
  and confusion matrix (classification), or R²/MAE/RMSE (regression)
- Overfitting check (train vs. test score gap)
- Permutation importance to explain which features matter
- Download the trained pipeline (`.joblib`) and predictions (`.csv`)
- Auto-generated `predict.py` script to run the model outside the app

## Tech stack

Python, Streamlit, pandas, NumPy, scikit-learn, XGBoost, joblib.
No paid APIs.

## Project structure
TrustML/
├── app.py # Streamlit UI
├── analyzer.py # Dataset analysis, problem-type detection
├── checks.py # Leakage detection, imbalance/distribution checks
├── preprocessing.py # Cleaning, ColumnTransformer, Pipeline, feature selection
├── models.py # Model registry, cross-validation comparison
├── tuning.py # RandomizedSearchCV hyperparameter tuning
├── evaluation.py # Test-set evaluation, overfitting check, importance
├── predictor.py # Model save/load, prediction on new data, inference script
├── requirements.txt
├── sample_data/ # Example CSVs to try
└── .streamlit/config.toml


## Running locally

```bash
git clone https://github.com/<your-username>/TrustML.git
cd TrustML
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

## How to use

1. Upload a CSV (try one from `sample_data/`).
2. Pick the target column in the **Target** tab; review the detected
   problem type and class balance / target distribution.
3. Check the **Warnings** tab for data-quality and leakage issues.
4. Go to **Train** to compare baseline models.
5. Go to **Tune** to hyperparameter-tune the top models.
6. Go to **Evaluate** to see honest test-set performance and feature
   importance.
7. Go to **Predict** to download the trained model, run it on new
   data, and download a ready-to-use `predict.py` script.

## Using the trained model in your own project

Download `trustml_model.joblib` and `predict.py` from the Predict tab
and keep them in the same folder.

```bash
python predict.py new_data.csv
```

```python
from predict import predict
import pandas as pd

new_data = pd.DataFrame([{"...": "..."}])
print(predict(new_data))
```

Note: only load `.joblib` files you trust — loading one can execute
code.

## Limitations (V1)

- Tabular CSV data only — no images, text, or audio
- Classification and regression only — no clustering or time series
- Best suited for small-to-medium datasets, not big data
- RandomizedSearchCV samples the hyperparameter space; it doesn't
  search exhaustively
- No automatic ordinal encoding — all categorical columns are one-hot
  encoded
- No built-in support for raw free text (e.g. email bodies) — such
  columns are flagged, not processed

## Future work

- FastAPI endpoint to serve the trained model over HTTP
- Ordinal encoding option for user-specified column order
- Support for free-text features via basic NLP (word counts, TF-IDF)
- Time-based train/test split for datasets with date columns
- Downloadable Markdown/HTML report of the full run

## Acknowledgements

Built as a learning project to practice the full ML workflow: EDA,
cleaning, leakage detection, preprocessing pipelines, model
comparison, hyperparameter tuning, evaluation, and deployment.
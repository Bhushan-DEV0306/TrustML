import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# ---------------------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------------------
def clean_dataset(df, target_column):
    """
    Clean the dataset and return (cleaned_df, decision_log).
    decision_log is a list of dicts explaining every action taken.
    """

    df = df.copy()
    decision_log = []

    # 1. Duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)

    if removed > 0:
        decision_log.append({
            "step": "remove_duplicates",
            "detail": f"Removed {removed} duplicate rows.",
        })

    # 2. Constant columns (only one unique value)
    constant_cols = [
        column for column in df.columns
        if column != target_column and df[column].nunique() <= 1
    ]

    if constant_cols:
        df = df.drop(columns=constant_cols)
        decision_log.append({
            "step": "remove_constant_columns",
            "detail": f"Removed constant columns: {constant_cols}",
        })

    # 3a. Rows that are mostly missing (more than 70% of columns empty)
    row_missing_percent = df.isnull().mean(axis=1) * 100
    rows_to_drop = row_missing_percent[row_missing_percent > 70].index

    if len(rows_to_drop) > 0:
        df = df.drop(index=rows_to_drop)
        decision_log.append({
            "step": "remove_missing_heavy_rows",
            "detail": (
                f"Removed {len(rows_to_drop)} rows with more than "
                "70% missing values."
            ),
        })

    # 3b. Columns that are mostly missing (more than 60% empty)
    col_missing_percent = df.isnull().mean() * 100
    heavy_missing_cols = col_missing_percent[
        col_missing_percent > 60
    ].index.tolist()

    if target_column in heavy_missing_cols:
        heavy_missing_cols.remove(target_column)

    if heavy_missing_cols:
        df = df.drop(columns=heavy_missing_cols)
        decision_log.append({
            "step": "remove_missing_heavy_columns",
            "detail": (
                f"Removed columns with more than 60% missing: "
                f"{heavy_missing_cols}"
            ),
        })

    # 4. ID-like columns (95%+ unique values)
    id_like_cols = []
    for column in df.columns:
        if column == target_column:
            continue

        non_null = df[column].dropna()
        if len(non_null) == 0:
            continue

        unique_percentage = non_null.nunique() / len(non_null) * 100
        if unique_percentage >= 95:
            id_like_cols.append(column)

    if id_like_cols:
        df = df.drop(columns=id_like_cols)
        decision_log.append({
            "step": "remove_id_like_columns",
            "detail": f"Removed ID-like columns: {id_like_cols}",
        })

    # 5. Rows with a missing target
    before = len(df)
    df = df.dropna(subset=[target_column])
    removed = before - len(df)

    if removed > 0:
        decision_log.append({
            "step": "remove_missing_target",
            "detail": f"Removed {removed} rows with missing target.",
        })

    return df, decision_log


# ---------------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------------
def split_dataset(df, target_column, problem_type, test_size=0.2, random_state=42):
    """
    Split into train/test. Stratified for classification so both
    sets keep a similar class balance.
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify = y if problem_type == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------
# PREPROCESSING (ColumnTransformer)
# ---------------------------------------------------------------
def build_preprocessor(X_train, max_categories=30):
    """
    Numeric columns: median imputation + scaling.
    Categorical columns: most-frequent imputation + one-hot encoding.
    Must be fit only on training data.
    """

    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()

    categorical_cols = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            max_categories=max_categories,
        )),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

    decision_log = {
        "numeric_columns": numeric_cols,
        "numeric_steps": ["median imputation", "standard scaling"],
        "categorical_columns": categorical_cols,
        "categorical_steps": [
            "most-frequent imputation",
            f"one-hot encoding (top {max_categories} categories, rest grouped as 'other')",
        ],
    }

    return preprocessor, decision_log


# ---------------------------------------------------------------
# TARGET ENCODING (classification only)
# ---------------------------------------------------------------
def encode_target(y_train, y_test, problem_type):
    if problem_type != "classification":
        return y_train, y_test, None

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    return y_train_encoded, y_test_encoded, encoder


# ---------------------------------------------------------------
# FULL PIPELINE (preprocessor + model)
# ---------------------------------------------------------------
def build_full_pipeline(preprocessor, model):
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return full_pipeline


# ---------------------------------------------------------------
# SMART NUMERIC CONVERSION (for values like "2 BHK", "2100-2850")
# ---------------------------------------------------------------
def try_parse_numeric_value(value):
    """
    Try to convert one messy value to a number.
    Handles: '2100', '2100-2850' (range -> average), '2 BHK' (first number).
    Returns float, or None if it can't be parsed.
    """

    if pd.isna(value):
        return None

    text = str(value).strip()

    try:
        return float(text)
    except ValueError:
        pass

    if "-" in text:
        parts = text.split("-")
        if len(parts) == 2:
            try:
                low = float(parts[0].strip())
                high = float(parts[1].strip())
                return (low + high) / 2
            except ValueError:
                pass

    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        return float(match.group(1))

    return None


def smart_numeric_conversion(df, min_success_rate=0.8):
    """
    Convert text columns to numeric when at least `min_success_rate`
    of their values can be parsed. Each log entry lists the converted
    column under the "columns" key.
    """

    df = df.copy()
    log = []

    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    for column in object_cols:
        non_null = df[column].dropna()
        if len(non_null) == 0:
            continue

        parsed = non_null.apply(try_parse_numeric_value)
        success_rate = parsed.notna().mean()

        if success_rate >= min_success_rate:
            df[column] = df[column].apply(try_parse_numeric_value)
            log.append({
                "step": "smart_numeric_conversion",
                "detail": (
                    f"Converted '{column}' to numeric "
                    f"({round(success_rate * 100, 1)}% of values parsed)."
                ),
                "columns": [column],
            })

    return df, log


# ---------------------------------------------------------------
# FEATURE SELECTION (Lasso)
# ---------------------------------------------------------------
class FeatureMaskSelector(BaseEstimator, TransformerMixin):
    """
    A simple pipeline step that keeps only the columns where mask is True.
    """

    def __init__(self, mask):
        self.mask = mask

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.mask]


def select_features_with_lasso(X_train_transformed, y_train, problem_type):
    """
    Use Lasso (regression) or L1-Logistic Regression (classification)
    to find which features have non-zero importance.
    Returns a boolean mask: True = keep this feature.
    """

    if problem_type == "regression":
        selector_model = Lasso(alpha=0.1, random_state=42)
    else:
        selector_model = LogisticRegression(
            penalty="l1", solver="liblinear", C=1.0, random_state=42
        )

    selector_model.fit(X_train_transformed, y_train)

    if problem_type == "regression":
        coefficients = selector_model.coef_
    else:
        coefficients = selector_model.coef_[0]

    important_features_mask = coefficients != 0
    return important_features_mask


def get_feature_names(preprocessor):
    """
    Output feature names from a fitted ColumnTransformer,
    e.g. ['numeric__age', 'categorical__city_Delhi', ...].
    """

    return preprocessor.get_feature_names_out()


def compare_with_without_feature_selection(
    X_train, y_train, problem_type, model, scoring=None
):
    """
    Compare CV score with all features vs only Lasso-selected features.
    """

    if scoring is None:
        scoring = "accuracy" if problem_type == "classification" else "r2"

    preprocessor, _ = build_preprocessor(X_train)

    full_pipeline = build_full_pipeline(preprocessor, model)
    full_scores = cross_val_score(
        full_pipeline, X_train, y_train, cv=5, scoring=scoring
    )

    preprocessor_fitted = build_preprocessor(X_train)[0]
    X_train_transformed = preprocessor_fitted.fit_transform(X_train)
    mask = select_features_with_lasso(X_train_transformed, y_train, problem_type)

    # If Lasso keeps too few features, skip the comparison
    if mask.sum() < 2:
        return {
            "full_score": round(full_scores.mean(), 4),
            "selected_score": None,
            "features_kept": int(mask.sum()),
            "features_total": len(mask),
            "use_selection": False,
            "note": "Too few features selected by Lasso; skipped comparison.",
        }

    selected_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(X_train)[0]),
        ("selector", FeatureMaskSelector(mask)),
        ("model", model),
    ])

    try:
        selected_scores = cross_val_score(
            selected_pipeline, X_train, y_train, cv=5, scoring=scoring
        )
        selected_mean = selected_scores.mean()
    except Exception:
        selected_mean = None

    if selected_mean is None or pd.isna(selected_mean):
        use_selection = False
        selected_score_display = None
    else:
        use_selection = selected_mean > full_scores.mean()
        selected_score_display = round(selected_mean, 4)

    return {
        "full_score": round(full_scores.mean(), 4),
        "selected_score": selected_score_display,
        "features_kept": int(mask.sum()),
        "features_total": len(mask),
        "use_selection": use_selection,
    }
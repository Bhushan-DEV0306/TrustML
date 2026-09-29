INFERENCE_TEMPLATE = r'''"""
TrustML inference script (auto-generated)

Put this file next to trustml_model.joblib.

Command line:  python predict.py new_data.csv
In your code:  from predict import predict

Trained with: scikit-learn __SKLEARN__, pandas __PANDAS__
(use the same versions if you can; install xgboost too if the model is XGBoost)

Only load model files you trust: joblib files can run code when loaded.
"""
import re
import sys

import joblib
import pandas as pd

MODEL_PATH = "trustml_model.joblib"

# The new data needs these columns:
# __FEATURES__


def _parse_number(value):
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
                return (float(parts[0].strip()) + float(parts[1].strip())) / 2
            except ValueError:
                pass

    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        return float(match.group(1))

    return None


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict(new_df, bundle=None):
    """Return predictions (original labels for classification)."""
    if bundle is None:
        bundle = load_model()

    df = new_df.copy()

    if bundle["target_column"] in df.columns:
        df = df.drop(columns=[bundle["target_column"]])

    for column in bundle["converted_columns"]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column].apply(_parse_number), errors="coerce"
            )

    missing = [c for c in bundle["feature_columns"] if c not in df.columns]
    if missing:
        raise ValueError(f"The new data is missing these columns: {missing}")

    predictions = bundle["pipeline"].predict(df[bundle["feature_columns"]])

    if bundle["target_encoder"] is not None:
        predictions = bundle["target_encoder"].inverse_transform(predictions)

    return predictions


if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1])
    data["Prediction"] = predict(data)
    print(data)
'''


def generate_inference_script(bundle):
    """Build a standalone predict.py for this specific trained model."""
    return (
         INFERENCE_TEMPLATE
        .replace("__SKLEARN__", sklearn.__version__)
        .replace("__PANDAS__", pd.__version__)
        .replace("__FEATURES__", ", ".join(bundle["feature_columns"]))
    )
import joblib
import sklearn
import pandas as pd

from preprocessing import try_parse_numeric_value

def save_model_bundle(
    pipeline, target_encoder, target_column, problem_type,
    feature_columns, path
):
    bundle = {
        "pipeline": pipeline,
        "target_encoder": target_encoder,
        "target_column": target_column,
        "problem_type": problem_type,
        "feature_columns": feature_columns,
    }

    joblib.dump(bundle, path)
    return path


def load_model_bundle(path):
    return joblib.load(path)

def save_model_bundle(
    pipeline, target_encoder, target_column, problem_type,
    feature_columns, converted_columns, path
):
    bundle = {
        "pipeline": pipeline,
        "target_encoder": target_encoder,
        "target_column": target_column,
        "problem_type": problem_type,
        "feature_columns": feature_columns,
        "converted_columns": converted_columns,
    }

    joblib.dump(bundle, path)
    return path

def predict_new_data(bundle, new_df):
    """
    Prepare new data the same way as the training data, predict,
    and return (result_df, notes).
    """

    df = new_df.copy()
    target_column = bundle["target_column"]
    feature_columns = bundle["feature_columns"]
    notes = []

    # 1. Target column is not needed for prediction
    if target_column in df.columns:
        df = df.drop(columns=[target_column])
        notes.append(f"Ignored the target column '{target_column}'.")

    # 2. Same text-to-number conversion as in training
    for column in bundle["converted_columns"]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column].apply(try_parse_numeric_value), errors="coerce"
            )

    # 3. Required columns must be present
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"The new data is missing these columns: {missing}")

    # 4. Keep only training columns, in training order
    extra = [c for c in df.columns if c not in feature_columns]
    if extra:
        notes.append(f"Ignored extra columns not used by the model: {extra}")

    X_new = df[feature_columns]

    # 5. Predict, then turn 0/1 back into the original labels
    predictions = bundle["pipeline"].predict(X_new)

    if bundle["target_encoder"] is not None:
        predictions = bundle["target_encoder"].inverse_transform(predictions)

    result = new_df.copy()
    result["Prediction"] = predictions

    return result, notes


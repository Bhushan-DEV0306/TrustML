

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, r2_score, mean_absolute_error,
    mean_squared_error,
)
import numpy as np
import pandas as pd

def get_test_predictions(best_model, X_train, y_train, X_test):
    """
    Fit the best model on the full training data, then predict on the
    held-out test set — data the model has never seen before.
    """

    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    return best_model, predictions





def evaluate_classification(y_test, predictions):
    """
    Detailed evaluation for classification: accuracy, precision,
    recall, f1, and confusion matrix.
    """

    return {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, average="weighted"), 4),
        "recall": round(recall_score(y_test, predictions, average="weighted"), 4),
        "f1": round(f1_score(y_test, predictions, average="weighted"), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }




def evaluate_regression(y_test, predictions):
    """
    Detailed evaluation for regression: R2, MAE, RMSE.
    """

    mse = mean_squared_error(y_test, predictions)

    return {
        "r2": round(r2_score(y_test, predictions), 4),
        "mae": round(mean_absolute_error(y_test, predictions), 4),
        "rmse": round(np.sqrt(mse), 4),
    }

#Checking Overfitting
def check_overfitting(fitted_model, X_train, y_train, test_score, scoring_func):
    """
    Compare train score vs test score to flag potential overfitting.
    scoring_func: a function like accuracy_score or r2_score.
    """

    train_predictions = fitted_model.predict(X_train)
    train_score = round(scoring_func(y_train, train_predictions), 4)

    gap = round(train_score - test_score, 4)

    if gap > 0.15:
        severity = "high"
        message = (
            f"Train score ({train_score}) is much higher than test score "
            f"({test_score}). This suggests overfitting."
        )
    elif gap > 0.05:
        severity = "medium"
        message = (
            f"Train score ({train_score}) is somewhat higher than test "
            f"score ({test_score}). Some overfitting may be present."
        )
    else:
        severity = "low"
        message = (
            f"Train score ({train_score}) and test score ({test_score}) "
            "are close. The model generalizes well."
        )

    return {
        "train_score": train_score,
        "test_score": test_score,
        "gap": gap,
        "severity": severity,
        "message": message,
    }
from sklearn.inspection import permutation_importance


def get_permutation_importance(fitted_model, X_test, y_test, scoring=None, n_repeats=10):
    """
    Compute permutation importance on the test set.
    Returns a DataFrame sorted by importance.
    """

    result = permutation_importance(
        fitted_model, X_test, y_test,
        scoring=scoring, n_repeats=n_repeats, random_state=42
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": result.importances_mean,
    })

    importance_df = importance_df.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)

    return importance_df
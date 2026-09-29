import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import cross_val_score
from preprocessing import build_preprocessor, build_full_pipeline


def get_models(problem_type):
    """
    Return a dictionary of {name: model} for the given problem type.
    Always includes a Dummy baseline first.
    """

    if problem_type == "classification":
        return {
            "Dummy": DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
            "KNN": KNeighborsClassifier(),
        }

    elif problem_type == "regression":
        return {
            "Dummy": DummyRegressor(strategy="mean"),
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "XGBoost": XGBRegressor(random_state=42),
            "KNN": KNeighborsRegressor(),
        }

    else:
        raise ValueError("problem_type must be classification or regression.")


def compare_models(X_train, y_train, problem_type, scoring=None):
    """
    Cross-validate every baseline model and return a comparison table,
    sorted best-to-worst by the chosen metric.
    """

    if scoring is None:
        scoring = "accuracy" if problem_type == "classification" else "r2"

    models = get_models(problem_type)
    preprocessor, _ = build_preprocessor(X_train)

    results = []

    for name, model in models.items():
        pipeline = build_full_pipeline(preprocessor, model)

        scores = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring=scoring
        )
        results.append({
            "Model": name,
            f"Mean {scoring.upper()}": round(scores.mean(), 4),
            "Std Dev": round(scores.std(), 4),
        })

    comparison_table = pd.DataFrame(results)
    comparison_table = comparison_table.sort_values(
        f"Mean {scoring.upper()}", ascending=False
    ).reset_index(drop=True)

    return comparison_table
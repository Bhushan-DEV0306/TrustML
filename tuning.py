
from sklearn.model_selection import RandomizedSearchCV
from preprocessing import build_preprocessor, build_full_pipeline
import pandas as pd 
def get_search_space(model_name, problem_type):
    """
  
      Return the hyperparameter search space for a given model name.
    Returns an empty dict if there's nothing meaningful to tune
    (e.g. Dummy, Linear/Logistic Regression with defaults).
    """

    search_spaces = {
        "Decision Tree": {
            "model__max_depth": [3, 5, 7, 10, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "Random Forest": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [5, 10, 15, None],
            "model__min_samples_split": [2, 5, 10],
        },
        "Gradient Boosting": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [2, 3, 5],
        },
        "KNN": {
            "model__n_neighbors": [3, 5, 7, 9, 11],
        },
        "XGBoost": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [2, 3, 5],
        },
    }

    return search_spaces.get(model_name, {})





def tune_model(model, model_name, X_train, y_train, problem_type, scoring=None, n_iter=10):
    search_space = get_search_space(model_name, problem_type)

    if not search_space:
        # Yahan bhi pipeline banao, taaki preprocessing ho
        preprocessor, _ = build_preprocessor(X_train)
        pipeline = build_full_pipeline(preprocessor, model)
        pipeline.fit(X_train, y_train)

        return {
            "model_name": model_name,
            "best_model": pipeline,
            "best_score": None,
            "best_params": {},
            "tuned": False,
        }

    if scoring is None:
        scoring = "accuracy" if problem_type == "classification" else "r2"

    preprocessor, _ = build_preprocessor(X_train)
    pipeline = build_full_pipeline(preprocessor, model)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=search_space,
        n_iter=n_iter,
        cv=5,
        scoring=scoring,
        random_state=42,
    )
    search.fit(X_train, y_train)

    return {
        "model_name": model_name,
        "best_model": search.best_estimator_,
        "best_score": round(search.best_score_, 4),
        "best_params": search.best_params_,
        "tuned": True,
    }

def tune_top_models(comparison_table, models_dict, X_train, y_train, problem_type, scoring=None, top_n=3):
    candidates = comparison_table[comparison_table["Model"] != "Dummy"]
    top_models = candidates.head(top_n)["Model"].tolist()

    score_col = comparison_table.columns[1]

    results = []

    for model_name in top_models:
        model = models_dict[model_name]

        untuned_score = comparison_table.loc[
            comparison_table["Model"] == model_name, score_col
        ].values[0]

        tune_result = tune_model(
            model, model_name, X_train, y_train, problem_type, scoring
        )

        if tune_result["tuned"]:
            tuned_score = tune_result["best_score"]
        else:
            tuned_score = untuned_score

        results.append({
            "Model": model_name,
            "Untuned Score": untuned_score,
            "Tuned Score": tuned_score,
            "Improvement": round(tuned_score - untuned_score, 4),
            "Best Params": tune_result["best_params"],
            "Best Model Object": tune_result["best_model"],
        })

    tuning_table = pd.DataFrame(results)
    tuning_table = tuning_table.sort_values(
        "Tuned Score", ascending=False
    ).reset_index(drop=True)

    return tuning_table
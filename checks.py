import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

def check_class_imbalance(df, target_column):
    """
    Check how balanced the classes of a classification target are.
    Returns a dictionary; it never prints anything.
    """

    target = df[target_column].dropna()
    missing_in_target = int(df[target_column].isnull().sum())

    if len(target) == 0:
        raise ValueError("The target column has no values.")

    class_counts = target.value_counts()

    if len(class_counts) < 2:
        raise ValueError("The target has only one class.")

    class_percentages = (class_counts / len(target) * 100).round(2)

    majority_class = str(class_counts.idxmax())
    minority_class = str(class_counts.idxmin())
    majority_count = int(class_counts.max())
    minority_count = int(class_counts.min())

    ratio = round(majority_count / minority_count, 2)

    if ratio < 3:
        severity = "balanced"
        message = "Classes are reasonably balanced."
        recommended_metric = "accuracy or f1"

    elif ratio < 9:
        severity = "moderate"
        message = (
            "Moderate class imbalance detected. "
            "Accuracy may be misleading."
        )
        recommended_metric = "f1"

    else:
        severity = "severe"
        message = (
            "Severe class imbalance detected. "
            "A model that always predicts the biggest class "
            "would still get a high accuracy."
        )
        recommended_metric = "f1 or recall"

    class_table = pd.DataFrame({
        "Class": class_counts.index.astype(str),
        "Count": class_counts.values,
        "Percentage": class_percentages.values,
    })

    return {
        "class_table": class_table,
        "majority_class": majority_class,
        "minority_class": minority_class,
        "ratio": ratio,
        "severity": severity,
        "message": message,
        "recommended_metric": recommended_metric,
        "small_class_warning": minority_count < 10,
        "missing_in_target": missing_in_target,
    }


def check_target_distribution(df, target_column):
    """
    Check distribution of a regression target.
    """

    target = df[target_column].dropna()

    if len(target) == 0:
        raise ValueError("Target column is empty.")

    if target.nunique() < 2:
        raise ValueError("All values in the target are the same.")

    issues = []

    # 1. Skewness
    skewness = round(target.skew(), 2)

    if skewness > 1:
        issues.append(
            "Target is right-skewed (a long tail of large values). "
            "A log transform may help."
        )

    elif skewness < -1:
        issues.append(
            "Target is left-skewed (a long tail of small values)."
        )

    # 2. Outliers using IQR method
    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_count = ((target < lower) | (target > upper)).sum()

    outlier_percentage = round(
        outlier_count / len(target) * 100,
        2
    )

    if outlier_percentage > 5:
        issues.append(
            f"{outlier_percentage}% of the values are outliers."
        )

    # 3. One value dominating the target
    top_value_percentage = round(
        target.value_counts(normalize=True).max() * 100,
        2
    )

    if top_value_percentage > 30:
        issues.append(
            f"One single value makes up {top_value_percentage}% "
            "of the target."
        )

    # 4. Severity
    if len(issues) == 0:
        severity = "balanced"

    elif len(issues) == 1:
        severity = "moderate"

    else:
        severity = "severe"

    if issues:
        message = " ".join(issues)

    else:
        message = "Target distribution looks fine."

    # 5. Metric suggestion
    if outlier_percentage > 5:
        recommended_metric = "MAE"

    else:
        recommended_metric = "RMSE"

    return {
        "skewness": skewness,
        "outlier_percentage": outlier_percentage,
        "top_value_percentage": top_value_percentage,
        "severity": severity,
        "message": message,
        "recommended_metric": recommended_metric,
    }


def check_data_quality(df, target_column):
    """
    Check general data-quality problems.
    """

    warnings = []

    # 1. Very few rows
    if len(df) < 100:
        warnings.append({
            "check": "few_rows",
            "severity": "high",
            "message": (
                f"Only {len(df)} rows. Results may be unreliable "
                "and cross-validation scores will vary a lot."
            ),
            "columns": [],
        })

    # 2. Columns with many missing values
    missing_percent = df.isnull().mean() * 100

    heavy_missing = missing_percent[
        missing_percent > 40
    ].index.tolist()

    if heavy_missing:
        warnings.append({
            "check": "heavy_missing",
            "severity": "medium",
            "message": (
                "These columns are more than 40% missing. "
                "Imputing them may add more guesswork than information."
            ),
            "columns": heavy_missing,
        })

    # 3. As many or more columns than rows
    if df.shape[1] >= df.shape[0]:
        warnings.append({
            "check": "too_many_columns",
            "severity": "high",
            "message": (
                "There are at least as many columns as rows. "
                "The model can easily overfit."
            ),
            "columns": [],
        })

    # 4. Missing values in target
    target_missing_count = int(
        df[target_column].isnull().sum()
    )

    if target_missing_count > 0:

        target_missing_percent = round(
            target_missing_count / len(df) * 100,
            2
        )

        warnings.append({
            "check": "target_missing",
            "severity": "low",
            "message": (
                f"{target_missing_count} rows "
                f"({target_missing_percent}%) have no target value. "
                "They will be removed during cleaning."
            ),
            "columns": [target_column],
        })

    return warnings


def run_all_checks(df, target_column, problem_type):
    """
    Run every check and return one list of warnings.

    Format:
    {
        "check": ...,
        "severity": ...,
        "message": ...,
        "columns": [...]
    }
    """

    if problem_type not in ("classification", "regression"):
        raise ValueError(
            "problem_type must be classification or regression."
        )

    warnings = []

    # 1. General dataset checks
    warnings.extend(
        check_data_quality(df, target_column)
    )
    warnings.extend(
        check_single_feature_predictive_power(df, target_column, problem_type)
    )
    warnings.extend(
        check_target_correlation(df, target_column, problem_type)
    )
    
    warnings.extend(check_id_like_columns(df, target_column))
    warnings.extend(check_suspicious_column_names(df, target_column))
    warnings.extend(check_date_columns(df, target_column))

    # 2. Target checks
    severity_map = {
        "moderate": "medium",
        "severe": "high"
    }

    if problem_type == "classification":

        try:
            balance = check_class_imbalance(
                df,
                target_column
            )

        except ValueError as error:

            warnings.append({
                "check": "target_problem",
                "severity": "high",
                "message": str(error),
                "columns": [target_column],
            })

        else:

            if balance["severity"] in severity_map:

                warnings.append({
                    "check": "class_imbalance",
                    "severity": severity_map[
                        balance["severity"]
                    ],
                    "message": (
                        f"{balance['message']} "
                        f"Suggested metric: "
                        f"{balance['recommended_metric']}."
                    ),
                    "columns": [target_column],
                })

            if balance["small_class_warning"]:

                warnings.append({
                    "check": "small_class",
                    "severity": "medium",
                    "message": (
                        "The smallest class has fewer than 10 rows. "
                        "Cross-validation may be unstable."
                    ),
                    "columns": [target_column],
                })

    else:

        try:
            dist = check_target_distribution(
                df,
                target_column
            )

        except ValueError as error:

            warnings.append({
                "check": "target_problem",
                "severity": "high",
                "message": str(error),
                "columns": [target_column],
            })

        else:

            if dist["severity"] in severity_map:

                warnings.append({
                    "check": "target_distribution",
                    "severity": severity_map[
                        dist["severity"]
                    ],
                    "message": (
                        f"{dist['message']} "
                        f"Suggested metric: "
                        f"{dist['recommended_metric']}."
                    ),
                    "columns": [target_column],
                })

    # 3. Most serious warnings first
    order = {
        "high": 0,
        "medium": 1,
        "low": 2
    }

    warnings.sort(
        key=lambda w: order[w["severity"]]
    )

    return warnings

"""
Leakage detection
"""
def check_target_correlation(df, target_column, problem_type):
    warnings = []

    if problem_type != "regression":
        return warnings

    num_cols = df.drop(columns=[target_column]).select_dtypes(include="number").columns
    corr = df[num_cols].corrwith(df[target_column])

    for column, value in corr.items():
        if pd.notna(value) and abs(value) >= 0.95:
            warnings.append({
                "check": "high_correlation",
                "severity": "high",
                "message": (
                    f"'{column}' has {round(value, 3)} correlation with "
                    "the target. This could be leakage."
                ),
                "columns": [column],
            })

    return warnings

"""
Signal 2 ka idea: har feature ko akela le ke ek chhota decision tree train karo, 
jo sirf usi ek feature se target predict karne ki koshish kare. Agar woh tree bhi bahut accha score de (jaise 98%+), 
to woh feature akela hi "cheat sheet" ban raha hai — real features aksar akele itna accha nahi karte.
"""
def check_single_feature_predictive_power(df, target_column, problem_type):
    warnings = []

    target = df[target_column].dropna()
    feature_df = df.loc[target.index]

    for column in df.columns:
        if column == target_column:
            continue
        feature = feature_df[[column]].copy()

        if pd.api.types.is_numeric_dtype(feature[column]):
            feature[column] = feature[column].fillna(feature[column].median())
        else:
            feature[column] = feature[column].astype("category").cat.codes


        if problem_type == "classification":
            model = DecisionTreeClassifier(max_depth=3, random_state=42)
            scoring = "accuracy"
        else:
            model = DecisionTreeRegressor(max_depth=3, random_state=42)
            scoring = "r2"
        
        try:
            scores = cross_val_score(model, feature, target, cv=3, scoring=scoring)
            mean_score = scores.mean()
        except Exception:
            continue

        if mean_score >= 0.95:
            warnings.append({
                "check": "single_feature_predictive",
                "severity": "high",
                "message": (
                    f"'{column}' alone predicts the target with "
                    f"{round(mean_score, 3)} {scoring}. This looks like leakage."
                ),
                "columns": [column],
            })
    return warnings

"""
for columns that have all values differnt .

"""
def check_id_like_columns(df, target_column):
    warnings = []

    for column in df.columns:
        if column == target_column:
            continue

        non_null = df[column].dropna()
        if len(non_null) == 0:
            continue

        unique_percentage = round(non_null.nunique() / len(non_null) * 100, 2)

        if unique_percentage >= 95:
                 avg_length = non_null.astype(str).str.len().mean()

                 if avg_length > 30:
                     warnings.append({
                    "check": "free_text_column",
                    "severity": "medium",
                    "message": (
                        f"'{column}' looks like free text (average "
                        f"{round(avg_length)} characters). This tool "
                        "handles tabular data, not raw text — extract "
                        "features (word count, keywords) first, or drop "
                        "this column."
                    ),
                    "columns": [column],
                })
                 else:
                     warnings.append({
                    "check": "id_like_column",
                    "severity": "medium",
                    "message": (
                        f"'{column}' has a different value in "
                        f"{unique_percentage}% of rows. It looks like an ID, "
                        "not a useful feature."
                    ),
                    "columns": [column],
                })
    return warnings

"""
Signal 4:Suspicious column names
"""
def check_suspicious_column_names(df, target_column):
    warnings = []

    suspicious_keywords = [
        "outcome", "result", "final", "status",
        "label", "prediction", "predicted",
    ]

    for column in df.columns:
        if column == target_column:
            continue

        for keyword in suspicious_keywords:
            if keyword in column.lower():
                warnings.append({
                    "check": "suspicious_column_name",
                    "severity": "low",
                    "message": (
                        f"'{column}' has a name that often indicates "
                        "leakage. Double-check it's not derived from "
                        "the target."
                    ),
                    "columns": [column],
                })
                break  

    return warnings

""" 
singnal 4:Date and Time 
"""
def check_date_columns(df, target_column):
    warnings = []
    date_keywords = ["date", "time", "year", "month", "day"]

    for column in df.columns:
        if column == target_column:
            continue

        is_date_dtype = pd.api.types.is_datetime64_any_dtype(df[column])
        name_looks_like_date = any(
            keyword in column.lower() for keyword in date_keywords
        )

        if is_date_dtype or name_looks_like_date:
            warnings.append({
                "check": "date_column",
                "severity": "low",
                "message": (
                    f"'{column}' looks like a date/time column. "
                    "A random train-test split could leak future "
                    "information into training. Consider a "
                    "time-based split."
                ),
                "columns": [column],
            })

    return warnings


    
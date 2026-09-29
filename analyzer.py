import pandas as pd


def analyze_dataset(df):

    analysis = {}


    # 1. BASIC DATASET INFORMATION


    analysis["rows"] = df.shape[0]
    analysis["columns"] = df.shape[1]

    # 2. DATA TYPES
  

    analysis["dtypes"] = df.dtypes

  
    # 3. NUMERIC COLUMNS


    analysis["numeric_columns"] = df.select_dtypes(
        include="number"
    ).columns.tolist()

    
    # 4. OBJECT / TEXT COLUMNS
    

    analysis["object_columns"] = df.select_dtypes(
        include="object"
    ).columns.tolist()

  
    # 5. UNIQUE VALUES
    

    analysis["unique_values"] = df.nunique()

  
    # 6. MISSING VALUES
    

    analysis["missing_values"] = df.isnull().sum()

    analysis["missing_percentage"] = (
        df.isnull().mean() * 100
    )

    
    # 7. DUPLICATES
   

    analysis["duplicates"] = df.duplicated().sum()

   
    # 8. OBJECT COLUMN INSPECTION
   

    object_analysis = []

    for column in analysis["object_columns"]:

        unique_count = df[column].nunique()

        total_count = len(df)

        unique_percentage = (
            unique_count / total_count
        ) * 100

        object_analysis.append({
            "column": column,
            "dtype": str(df[column].dtype),
            "unique_values": unique_count,
            "unique_percentage": round(
                unique_percentage, 2
            )
        })

    analysis["object_analysis"] = pd.DataFrame(
        object_analysis
    )

    return analysis


#""" Target Selection"""


def detect_problem_type(df, target_column):

    """
    Guess whether the target column is a classification or
    regression problem, and explain why.
    """

    target = df[target_column].dropna()

    if len(target) == 0:
        raise ValueError("The target column has no values.")

    unique_count = target.nunique()

    if unique_count < 2:
        raise ValueError(
            "The target column has only one unique value, "
            "so there is nothing to predict."
        )


    is_bool = pd.api.types.is_bool_dtype(target)
    is_numeric = pd.api.types.is_numeric_dtype(target) and not is_bool

    if not is_numeric:
        problem_type = "classification"
        reason = "Target contains text, categories or True/False values."




    elif unique_count <= 10:
        problem_type = "classification"
        reason = (
            f"Target is numeric but has only {unique_count} "
            "unique values, so it looks like classes."
        )
    else:
        problem_type = "regression"
        reason = (
            f"Target is numeric with {unique_count} unique "
            "values, so it looks like a continuous number."
        )

    return {
        "target": target_column,
        "problem_type": problem_type,
        "unique_values": unique_count,
        "missing_in_target": int(df[target_column].isnull().sum()),
        "reason": reason,
    }
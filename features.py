
import numpy as np
import pandas as pd
import streamlit as st
from preprocessing import clean_dataset, split_dataset, encode_target
from models import compare_models
from analyzer import analyze_dataset, detect_problem_type
from checks import (
    check_class_imbalance,
    check_target_distribution,
    run_all_checks,
)

st.set_page_config(page_title="TrustML", page_icon="🤖", layout="wide")

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🤖 TrustML")
    st.caption("Trustworthy, explainable AutoML")
    st.divider()

    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    st.divider()
    st.markdown("**Pipeline**")
    st.markdown(
        "1. ✅ Analyze data\n"
        "2. 🟡 Checks & warnings\n"
        "3. ⬜ Clean & preprocess\n"
        "4. ⬜ Train & compare models\n"
        "5. ⬜ Tune & explain\n"
        "6. ⬜ Predict & download"
    )

# ---------- Header ----------
st.title("Explainable AutoML Platform")
st.caption("Understand your data before you trust any model.")

# ---------- Welcome screen ----------
if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to begin.")

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("### 🔍 Explainable")
            st.write("See every preprocessing step and why it was chosen.")
    with c2:
        with st.container(border=True):
            st.markdown("### 🛡️ Leakage-aware")
            st.write("Suspicious features are flagged before training.")
    with c3:
        with st.container(border=True):
            st.markdown("### 🎯 Metric-aware")
            st.write("Models are picked using the metric that fits your data.")
    st.stop()

# ---------- Read the file ----------
try:
    df = pd.read_csv(uploaded_file)
    analysis = analyze_dataset(df)
except Exception as error:
    st.error(f"Could not read this file: {error}")
    st.stop()

# ---------- Top metrics ----------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{analysis['rows']:,}")
m2.metric("Columns", analysis["columns"])
m3.metric("Duplicate rows", int(analysis["duplicates"]))
m4.metric(
    "Columns with missing",
    int((analysis["missing_values"] > 0).sum()),
)

# These are filled in by the Target tab and used by the Warnings tab.
# No st.stop() inside the tabs, so every tab always renders.
target_column = None
final_type = None

# ---------- Tabs ----------
tab_overview, tab_columns, tab_target, tab_warnings, tab_train = st.tabs(
    ["📋 Overview", "🔎 Columns", "🎯 Target", "⚠️ Warnings", "🚀 Train"]
)

# ----- Tab 1: Overview -----
with tab_overview:
    st.subheader("Data preview")
    st.dataframe(df.head(10))

    st.subheader("Quick warnings")
    warnings_found = False

    if analysis["duplicates"] > 0:
        st.warning(f"{int(analysis['duplicates'])} duplicate rows found.")
        warnings_found = True

    if not analysis["object_analysis"].empty:
        id_like = analysis["object_analysis"][
            analysis["object_analysis"]["unique_percentage"] >= 95
        ]["column"].tolist()
        if id_like:
            st.warning(
                f"ID-like columns (almost every value unique): {id_like}. "
                "These usually should not be used as features."
            )
            warnings_found = True

    constant_cols = analysis["unique_values"][
        analysis["unique_values"] <= 1
    ].index.tolist()
    if constant_cols:
        st.warning(f"Constant columns (no information): {constant_cols}")
        warnings_found = True

    if not warnings_found:
        st.success("No basic data-quality warnings.")

# ----- Tab 2: Columns -----
with tab_columns:
    st.subheader("Column summary")
    summary = pd.DataFrame({
        "dtype": analysis["dtypes"].astype(str),
        "unique_values": analysis["unique_values"],
        "missing": analysis["missing_values"],
        "missing_%": analysis["missing_percentage"].round(2),
    })
    st.dataframe(summary)

    left, right = st.columns(2)
    with left:
        st.markdown("**Numeric columns**")
        st.write(analysis["numeric_columns"] or "None")
    with right:
        st.markdown("**Text / categorical columns**")
        st.write(analysis["object_columns"] or "None")

    missing = analysis["missing_percentage"]
    missing = missing[missing > 0]
    if not missing.empty:
        st.subheader("Missing values (%)")
        st.bar_chart(missing)

    if not analysis["object_analysis"].empty:
        st.subheader("Text column inspection")
        st.dataframe(analysis["object_analysis"])

# ----- Tab 3: Target -----
with tab_target:
    st.subheader("Choose the target column")

    target_column = st.selectbox(
        "Which column do you want to predict?",
        options=df.columns,
        index=None,
        placeholder="Select the target column...",
    )

    if target_column is None:
        st.info("Select the column you want to predict to continue.")
    else:
        # An ID column is almost never a real prediction target
        target_values = df[target_column].dropna()
        if len(target_values) > 0 and target_values.nunique() == len(target_values):
            st.warning(
                f"'{target_column}' has a different value in every row. "
                "It looks like an ID column, not a real target."
            )

        try:
            result = detect_problem_type(df, target_column)
        except ValueError as error:
            st.error(str(error))
            result = None

        if result is not None:
            with st.container(border=True):
                st.markdown(f"**Detected:** {result['problem_type'].title()}")
                st.caption(result["reason"])

            options = ["classification", "regression"]
            final_type = st.radio(
                "Confirm or change the problem type:",
                options,
                index=options.index(result["problem_type"]),
                horizontal=True,
            )

            st.success(
                f"Target: **{target_column}** | Problem type: **{final_type}**"
            )

            # ----- Classification: class balance -----
            if final_type == "classification":
                st.subheader("Class balance")

                try:
                    balance = check_class_imbalance(df, target_column)
                except ValueError as error:
                    st.error(str(error))
                else:
                    if balance["severity"] == "balanced":
                        st.success(balance["message"])
                    elif balance["severity"] == "moderate":
                        st.warning(balance["message"])
                    else:
                        st.error(balance["message"])

                    st.write(
                        f"Biggest class is **{balance['ratio']}x** larger "
                        f"than the smallest. Suggested metric: "
                        f"**{balance['recommended_metric']}**"
                    )

                    st.dataframe(balance["class_table"])
                    st.bar_chart(
                        balance["class_table"].set_index("Class")["Count"]
                    )

            # ----- Regression: target distribution -----
            else:
                st.subheader("Target distribution")

                try:
                    dist = check_target_distribution(df, target_column)
                except ValueError as error:
                    st.error(str(error))
                else:
                    if dist["severity"] == "balanced":
                        st.success(dist["message"])
                    elif dist["severity"] == "moderate":
                        st.warning(dist["message"])
                    else:
                        st.error(dist["message"])

                    d1, d2, d3 = st.columns(3)
                    d1.metric("Skewness", dist["skewness"])
                    d2.metric("Outliers (%)", dist["outlier_percentage"])
                    d3.metric(
                        "Most common value (%)", dist["top_value_percentage"]
                    )

                    st.write(
                        f"Suggested metric: **{dist['recommended_metric']}**"
                    )

                    counts, edges = np.histogram(
                        df[target_column].dropna(), bins=20
                    )
                    histogram = pd.DataFrame(
                        {"count": counts},
                        index=[round(float(e), 2) for e in edges[:-1]],
                    )
                    st.bar_chart(histogram)

# ----- Tab 4: Warnings -----
with tab_warnings:
    st.subheader("All checks in one place")

    if target_column is None or final_type is None:
        st.info("Choose a target column in the Target tab first.")
    else:
        found = run_all_checks(df, target_column, final_type)

        if not found:
            st.success("✓ No major issues found by the current checks.")
        else:
            high = sum(1 for w in found if w["severity"] == "high")
            medium = sum(1 for w in found if w["severity"] == "medium")
            low = sum(1 for w in found if w["severity"] == "low")

            s1, s2, s3 = st.columns(3)
            s1.metric("High", high)
            s2.metric("Medium", medium)
            s3.metric("Low", low)

            for w in found:
                text = w["message"]
                if w["columns"]:
                    text += f"  \n**Columns:** {', '.join(map(str, w['columns']))}"

                if w["severity"] == "high":
                    st.error(text)
                elif w["severity"] == "medium":
                    st.warning(text)
                else:
                    st.info(text)


# ----- Tab 5: Train -----
with tab_train:
    st.subheader("Train baseline models")

    if target_column is None or final_type is None:
        st.info("Choose a target column in the Target tab first.")
    else:
        if st.button("Train Models", type="primary"):
            progress_bar = st.progress(0, text="Cleaning data...")

            cleaned_df, clean_log = clean_dataset(df, target_column)
            progress_bar.progress(25, text="Splitting into train/test...")

            X_train, X_test, y_train, y_test = split_dataset(
                cleaned_df, target_column, final_type
            )
            progress_bar.progress(40, text="Encoding target...")

            y_train_enc, y_test_enc, target_encoder = encode_target(
                y_train, y_test, final_type
            )
            progress_bar.progress(
                50, text="Training and comparing models (this may take a moment)..."
            )

            comparison_table = compare_models(X_train, y_train_enc, final_type)
            progress_bar.progress(100, text="Done!")

            st.success("Training complete!")

            st.subheader("Cleaning steps applied")
            if clean_log:
                for entry in clean_log:
                    st.write(f"- {entry['detail']}")
            else:
                st.write("No cleaning steps were needed.")

            st.subheader("Model comparison")
            st.dataframe(comparison_table, use_container_width=True)

            best_model_name = comparison_table.iloc[0]["Model"]
            best_score_col = comparison_table.columns[1]
            best_score = comparison_table.iloc[0][best_score_col]
            st.info(
                f"Best model so far: **{best_model_name}** "
                f"with a mean {best_score_col} of **{best_score}**"
            )
           

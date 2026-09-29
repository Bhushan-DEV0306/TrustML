import io

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, r2_score

from analyzer import analyze_dataset, detect_problem_type
from checks import check_class_imbalance, check_target_distribution, run_all_checks
from preprocessing import (
    clean_dataset,
    split_dataset,
    encode_target,
    smart_numeric_conversion,
    compare_with_without_feature_selection,
)
from models import compare_models, get_models
from tuning import tune_top_models
from evaluation import (
    get_test_predictions, evaluate_classification, evaluate_regression,
    check_overfitting, get_permutation_importance,
)
from predictor import (
    save_model_bundle, load_model_bundle, predict_new_data,
    generate_inference_script,
)

st.set_page_config(page_title="TrustML", page_icon="🤖", layout="wide")

# ---------- Landing page gate ----------
if "app_started" not in st.session_state:
    st.session_state["app_started"] = False

if not st.session_state["app_started"]:
    st.markdown("# 🤖 **TrustML**")
    st.markdown("### **Trustworthy, Explainable AutoML — Understand your data before you trust any model.**")
    st.write("")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("## ✅ What makes it different")
            st.markdown(
                "- **Explainable**: every preprocessing step and model choice is shown, not hidden\n"
                "- **Leakage-aware**: flags suspicious features before training, not after\n"
                "- **Metric-aware**: lets you pick the metric that fits your data (F1, MAE, R², etc.)\n"
                "- **Honest evaluation**: always compares against a dummy baseline, never invents accuracy\n"
                "- **Smart tuning**: only tunes the top models, not every model blindly\n"
                "- **Ready to use**: download the trained model plus a script to run it in your own project"
            )

    with right:
        with st.container(border=True):
            st.markdown("## ⚠️ Current limitations")
            st.markdown(
                "- Works with **tabular CSV data** only — no images, text, or audio\n"
                "- Supports **classification and regression** only, no clustering or time series\n"
                "- Best suited for **small to medium datasets**, not big data\n"
                "- Hyperparameter tuning uses a **randomized search**, not an exhaustive one\n"
                "- Free-tier hosting may be **slower** on large files or heavy tuning"
            )

    st.write("")
    st.write("")

    c1, c2, c3 = st.columns([2, 1, 2])
    with c2:
        if st.button("🚀 Try Now", type="primary", use_container_width=True):
            st.session_state["app_started"] = True
            st.rerun()

    st.divider()
    st.caption(
        "TrustML · Built with Streamlit and scikit-learn · "
        "An explainable AutoML platform for hackathons and learning projects"
    )

    st.stop()

# ---------- Session defaults ----------
target_column = None
final_type = None
trained = "comparison_table" in st.session_state
tuned = "tuning_table" in st.session_state
evaluated = "eval_results" in st.session_state

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🤖 TrustML")
    st.caption("Trustworthy, explainable AutoML")
    st.divider()

    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    st.divider()
    st.markdown("**Pipeline**")

    step1 = "✅" if uploaded_file is not None else "⬜"
    step2 = "✅" if st.session_state.get("target_column") else "⬜"
    step3 = "✅" if trained else "⬜"
    step4 = "✅" if tuned else "⬜"
    step5 = "✅" if evaluated else "⬜"
    step6 = "✅" if "model_bytes" in st.session_state else "⬜"

    st.markdown(
        f"{step1} Analyze data\n\n"
        f"{step2} Choose target & review warnings\n\n"
        f"{step3} Train & compare models\n\n"
        f"{step4} Tune top models\n\n"
        f"{step5} Evaluate & explain\n\n"
        f"{step6} Predict & download"
    )

    if evaluated:
        st.divider()
        st.markdown("**🏆 Final model**")
        st.markdown(f"{st.session_state['eval_model_name']}")
        key_metric = (
            "accuracy" if st.session_state.get("final_type") == "classification" else "r2"
        )
        st.markdown(f"`{st.session_state['eval_results'].get(key_metric)}`")
    elif tuned:
        st.divider()
        best_row = st.session_state["tuning_table"].iloc[0]
        st.markdown("**🏆 Current best (tuned)**")
        st.markdown(f"{best_row['Model']}")
        st.markdown(f"`{best_row['Tuned Score']}`")
    elif trained:
        st.divider()
        best_row = st.session_state["comparison_table"].iloc[0]
        st.markdown("**🏆 Current best**")
        st.markdown(f"{best_row['Model']}")
        st.markdown(f"`{best_row.iloc[1]}`")

# ---------- Header ----------
st.title("Explainable AutoML Platform")
st.caption("Understand your data before you trust any model.")

# ---------- Welcome screen (no file yet) ----------
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
m4.metric("Columns with missing", int((analysis["missing_values"] > 0).sum()))

st.divider()

# ---------- Tabs ----------
(
    tab_overview, tab_columns, tab_target, tab_warnings,
    tab_train, tab_tune, tab_evaluate, tab_predict,
) = st.tabs(
    ["📋 Overview", "🔎 Columns", "🎯 Target", "⚠️ Warnings",
     "🚀 Train", "🎯 Tune", "📈 Evaluate", "🔮 Predict"]
)

# ===== Tab 1: Overview =====
with tab_overview:
    st.subheader("Data preview")
    st.dataframe(df.head(10), width="stretch")

    st.subheader("Quick warnings")
    warnings_found = False

    if analysis["duplicates"] > 0:
        st.warning(f"⚠️ {int(analysis['duplicates'])} duplicate rows found.")
        warnings_found = True

    if not analysis["object_analysis"].empty:
        id_like = analysis["object_analysis"][
            analysis["object_analysis"]["unique_percentage"] >= 95
        ]["column"].tolist()
        if id_like:
            st.warning(
                f"⚠️ ID-like columns (almost every value unique): {id_like}. "
                "These usually should not be used as features."
            )
            warnings_found = True

    constant_cols = analysis["unique_values"][
        analysis["unique_values"] <= 1
    ].index.tolist()
    if constant_cols:
        st.warning(f"⚠️ Constant columns (no information): {constant_cols}")
        warnings_found = True

    if not warnings_found:
        st.success("✅ No basic data-quality warnings.")

# ===== Tab 2: Columns =====
with tab_columns:
    st.subheader("Column summary")
    summary = pd.DataFrame({
        "dtype": analysis["dtypes"].astype(str),
        "unique_values": analysis["unique_values"],
        "missing": analysis["missing_values"],
        "missing_%": analysis["missing_percentage"].round(2),
    })
    st.dataframe(summary, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**🔢 Numeric columns**")
        st.write(analysis["numeric_columns"] or "None")
    with right:
        st.markdown("**🔤 Text / categorical columns**")
        st.write(analysis["object_columns"] or "None")

    missing = analysis["missing_percentage"]
    missing = missing[missing > 0]
    if not missing.empty:
        st.subheader("Missing values (%)")
        st.bar_chart(missing)

    if not analysis["object_analysis"].empty:
        with st.expander("🔎 Text column inspection"):
            st.dataframe(analysis["object_analysis"], width="stretch")

# ===== Tab 3: Target =====
with tab_target:
    st.subheader("Choose the target column")

    target_column = st.selectbox(
        "Which column do you want to predict?",
        options=df.columns,
        index=None,
        placeholder="Select the target column...",
    )
    st.session_state["target_column"] = target_column

    if target_column is None:
        st.info("Select the column you want to predict to continue.")
    else:
        target_values = df[target_column].dropna()
        if len(target_values) > 0 and target_values.nunique() == len(target_values):
            st.warning(
                f"⚠️ '{target_column}' has a different value in every row. "
                "It looks like an ID column, not a real target."
            )

        try:
            result = detect_problem_type(df, target_column)
        except ValueError as error:
            st.error(str(error))
            result = None

        if result is not None:
            with st.container(border=True):
                col_a, col_b = st.columns([1, 3])
                col_a.metric("Detected type", result["problem_type"].title())
                col_b.write(result["reason"])

            options = ["classification", "regression"]
            final_type = st.radio(
                "Confirm or change the problem type:",
                options,
                index=options.index(result["problem_type"]),
                horizontal=True,
            )
            st.session_state["final_type"] = final_type

            st.success(
                f"🎯 Target: **{target_column}** | Problem type: **{final_type}**"
            )

            st.divider()

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

                    c1, c2 = st.columns([2, 3])
                    with c1:
                        st.dataframe(balance["class_table"], width="stretch")
                    with c2:
                        st.bar_chart(balance["class_table"].set_index("Class")["Count"])

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
                    d3.metric("Most common value (%)", dist["top_value_percentage"])

                    st.write(f"Suggested metric: **{dist['recommended_metric']}**")

                    counts, edges = np.histogram(df[target_column].dropna(), bins=20)
                    histogram = pd.DataFrame(
                        {"count": counts},
                        index=[round(float(e), 2) for e in edges[:-1]],
                    )
                    st.bar_chart(histogram)

# Fall back to session values so later tabs work across reruns
target_column = target_column or st.session_state.get("target_column")
final_type = final_type or st.session_state.get("final_type")

# ===== Tab 4: Warnings =====
with tab_warnings:
    st.subheader("All checks in one place")

    if target_column is None or final_type is None:
        st.info("Choose a target column in the Target tab first.")
    else:
        found = run_all_checks(df, target_column, final_type)

        if not found:
            st.success("✅ No major issues found by the current checks.")
        else:
            high = sum(1 for w in found if w["severity"] == "high")
            medium = sum(1 for w in found if w["severity"] == "medium")
            low = sum(1 for w in found if w["severity"] == "low")

            s1, s2, s3 = st.columns(3)
            s1.metric("🔴 High", high)
            s2.metric("🟡 Medium", medium)
            s3.metric("🔵 Low", low)

            st.divider()

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

# ===== Tab 5: Train =====
with tab_train:
    st.subheader("Train baseline models")

    if target_column is None or final_type is None:
        st.info("Choose a target column in the Target tab first.")
    else:
        if final_type == "classification":
            metric_options = ["accuracy", "f1", "precision", "recall", "roc_auc"]
        else:
            metric_options = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]

        metric_labels = {
            "accuracy": "Accuracy",
            "f1": "F1 Score",
            "precision": "Precision",
            "recall": "Recall",
            "roc_auc": "ROC-AUC",
            "r2": "R² (variance explained)",
            "neg_mean_absolute_error": "MAE (mean absolute error)",
            "neg_root_mean_squared_error": "RMSE (root mean squared error)",
        }

        chosen_metric = st.selectbox(
            "Which metric should decide the best model?",
            options=metric_options,
            format_func=lambda x: metric_labels[x],
        )
        st.session_state["chosen_metric"] = chosen_metric

        if st.button("🚀 Train Models", type="primary"):
            progress_bar = st.progress(0, text="Cleaning data...")

            smart_df, smart_log = smart_numeric_conversion(df)
            cleaned_df, clean_log = clean_dataset(smart_df, target_column)
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
            comparison_table = compare_models(
                X_train, y_train_enc, final_type, scoring=chosen_metric
            )
            progress_bar.progress(100, text="Done!")

            st.session_state["comparison_table"] = comparison_table
            st.session_state["clean_log_display"] = smart_log + clean_log
            st.session_state["X_train"] = X_train
            st.session_state["y_train_enc"] = y_train_enc
            st.session_state["X_test"] = X_test
            st.session_state["y_test_enc"] = y_test_enc

            # Needed later to prepare new data exactly like the training data
            st.session_state["target_encoder"] = target_encoder
            st.session_state["feature_columns"] = list(X_train.columns)
            st.session_state["converted_columns"] = [
                column for entry in smart_log for column in entry["columns"]
            ]

            # Clear stale results from a previous dataset/run
            st.session_state.pop("tuning_table", None)
            st.session_state.pop("eval_results", None)
            st.session_state.pop("model_bytes", None)

            st.success("✅ Training complete!")

        if "comparison_table" in st.session_state:
            st.divider()

            with st.expander("🧹 Cleaning steps applied", expanded=False):
                logs = st.session_state.get("clean_log_display", [])
                if logs:
                    for entry in logs:
                        st.write(f"- {entry['detail']}")
                else:
                    st.write("No cleaning steps were needed.")

            st.subheader("📊 Model comparison")
            comparison_table = st.session_state["comparison_table"]

            best_model_name = comparison_table.iloc[0]["Model"]
            best_score_col = comparison_table.columns[1]
            best_score = comparison_table.iloc[0][best_score_col]
            dummy_score = comparison_table.loc[
                comparison_table["Model"] == "Dummy", best_score_col
            ].values[0]

            k1, k2, k3 = st.columns(3)
            k1.metric("🏆 Best model", best_model_name)
            k2.metric(best_score_col, best_score)
            k3.metric("vs. Dummy baseline", f"{best_score - dummy_score:+.4f}")

            st.dataframe(
                comparison_table.style.background_gradient(
                    subset=[best_score_col], cmap="Blues"
                ),
                width="stretch",
            )

            st.divider()
            with st.expander("🔬 Try feature selection (optional, may take a few minutes)"):
                st.caption(
                    "Uses Lasso to find which features are genuinely useful, "
                    "then compares scores with vs without them. Only "
                    "recommended if it genuinely improves the score."
                )
                if st.button("Run feature selection check"):
                    with st.spinner(
                        "Running Lasso feature selection (this can take a few minutes)..."
                    ):
                        models_dict = get_models(final_type)
                        best_model_obj = models_dict[best_model_name]

                        fs_report = compare_with_without_feature_selection(
                            st.session_state["X_train"],
                            st.session_state["y_train_enc"],
                            final_type,
                            best_model_obj,
                        )

                    c1, c2 = st.columns(2)
                    c1.metric("Score (all features)", fs_report["full_score"])
                    c2.metric(
                        "Score (selected features)",
                        fs_report["selected_score"]
                        if fs_report.get("selected_score") is not None
                        else "N/A",
                    )

                    st.write(
                        f"Kept {fs_report['features_kept']} out of "
                        f"{fs_report['features_total']} features."
                    )

                    if fs_report["use_selection"]:
                        st.success(
                            "✅ Feature selection genuinely improved the score — "
                            "recommended to use it."
                        )
                    else:
                        st.info(
                            "ℹ️ Feature selection did not improve the score — "
                            "keeping all features is better here."
                        )

# ===== Tab 6: Tune =====
with tab_tune:
    st.subheader("Tune the top models")

    if "comparison_table" not in st.session_state:
        st.info("Train models in the Train tab first.")
    else:
        st.caption(
            "Takes the top models (excluding Dummy) from the comparison table "
            "and tunes their hyperparameters using RandomizedSearchCV. "
            "This can take a few minutes."
        )

        top_n = st.slider("How many top models to tune?", min_value=1, max_value=5, value=3)

        if st.button("🎯 Tune Top Models", type="primary"):
            with st.spinner("Tuning hyperparameters (this may take a few minutes)..."):
                models_dict = get_models(final_type)
                chosen_metric = st.session_state.get("chosen_metric")

                tuning_table = tune_top_models(
                    st.session_state["comparison_table"],
                    models_dict,
                    st.session_state["X_train"],
                    st.session_state["y_train_enc"],
                    final_type,
                    scoring=chosen_metric,
                    top_n=top_n,
                )

            st.session_state["tuning_table"] = tuning_table
            st.session_state.pop("eval_results", None)
            st.session_state.pop("model_bytes", None)
            st.success("✅ Tuning complete!")

        if "tuning_table" in st.session_state:
            st.divider()

            tuning_table = st.session_state["tuning_table"]
            best_tuned = tuning_table.iloc[0]

            t1, t2, t3 = st.columns(3)
            t1.metric("🏆 Best tuned model", best_tuned["Model"])
            t2.metric("Tuned score", best_tuned["Tuned Score"])
            t3.metric("Improvement over untuned", f"{best_tuned['Improvement']:+.4f}")

            st.subheader("📊 Untuned vs Tuned comparison")
            st.dataframe(
                tuning_table[["Model", "Untuned Score", "Tuned Score", "Improvement"]]
                .style.background_gradient(subset=["Improvement"], cmap="Greens"),
                width="stretch",
            )

            with st.expander("🔧 Best hyperparameters found"):
                for _, row in tuning_table.iterrows():
                    st.markdown(f"**{row['Model']}**")
                    if row["Best Params"]:
                        st.json(row["Best Params"])
                    else:
                        st.write("No tuning was applied (no search space defined).")

# ===== Tab 7: Evaluate =====
with tab_evaluate:
    st.subheader("Final evaluation on held-out test data")

    if "tuning_table" not in st.session_state:
        st.info("Tune models in the Tune tab first.")
    else:
        if st.button("📈 Run Final Evaluation", type="primary"):
            with st.spinner("Evaluating the best model on the test set..."):
                tuning_table = st.session_state["tuning_table"]
                best_model = tuning_table.iloc[0]["Best Model Object"]
                best_model_name = tuning_table.iloc[0]["Model"]

                X_train_s = st.session_state["X_train"]
                y_train_s = st.session_state["y_train_enc"]
                X_test_s = st.session_state["X_test"]
                y_test_s = st.session_state["y_test_enc"]

                fitted_model, predictions = get_test_predictions(
                    best_model, X_train_s, y_train_s, X_test_s
                )

                if final_type == "classification":
                    eval_results = evaluate_classification(y_test_s, predictions)
                    main_score = eval_results["accuracy"]
                    scoring_func = accuracy_score
                else:
                    eval_results = evaluate_regression(y_test_s, predictions)
                    main_score = eval_results["r2"]
                    scoring_func = r2_score

                overfit_result = check_overfitting(
                    fitted_model, X_train_s, y_train_s, main_score, scoring_func
                )

                importance_table = get_permutation_importance(
                    fitted_model, X_test_s, y_test_s,
                    scoring="accuracy" if final_type == "classification" else "r2",
                )

                # Pack the final model + everything needed to prepare new data.
                # Kept in memory (not on disk) so it also works when deployed.
                model_buffer = io.BytesIO()
                save_model_bundle(
                    fitted_model,
                    st.session_state["target_encoder"],
                    target_column,
                    final_type,
                    st.session_state["feature_columns"],
                    st.session_state["converted_columns"],
                    model_buffer,
                )

            st.session_state["eval_results"] = eval_results
            st.session_state["overfit_result"] = overfit_result
            st.session_state["importance_table"] = importance_table
            st.session_state["eval_model_name"] = best_model_name
            st.session_state["model_bytes"] = model_buffer.getvalue()

            st.success("✅ Evaluation complete! You can now use the Predict tab.")

        if "eval_results" in st.session_state:
            st.divider()
            eval_results = st.session_state["eval_results"]
            overfit_result = st.session_state["overfit_result"]
            importance_table = st.session_state["importance_table"]

            st.caption(f"Model: **{st.session_state['eval_model_name']}**")

            # ----- Metrics -----
            if final_type == "classification":
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Accuracy", eval_results["accuracy"])
                e2.metric("Precision", eval_results["precision"])
                e3.metric("Recall", eval_results["recall"])
                e4.metric("F1 Score", eval_results["f1"])

                st.subheader("Confusion Matrix")
                cm = eval_results["confusion_matrix"]
                cm_df = pd.DataFrame(
                    cm,
                    index=[f"Actual: {i}" for i in range(cm.shape[0])],
                    columns=[f"Predicted: {i}" for i in range(cm.shape[1])],
                )
                st.dataframe(
                    cm_df.style.background_gradient(cmap="Blues"),
                    width="stretch",
                )
            else:
                e1, e2, e3 = st.columns(3)
                e1.metric("R²", eval_results["r2"])
                e2.metric("MAE", eval_results["mae"])
                e3.metric("RMSE", eval_results["rmse"])

            st.divider()

            # ----- Overfitting -----
            st.subheader("Overfitting check")
            if overfit_result["severity"] == "low":
                st.success(overfit_result["message"])
            elif overfit_result["severity"] == "medium":
                st.warning(overfit_result["message"])
            else:
                st.error(overfit_result["message"])

            o1, o2, o3 = st.columns(3)
            o1.metric("Train score", overfit_result["train_score"])
            o2.metric("Test score", overfit_result["test_score"])
            o3.metric("Gap", overfit_result["gap"])

            st.divider()

            # ----- Feature importance -----
            st.subheader("What matters most? (Permutation Importance)")
            st.caption(
                "Each feature is shuffled to see how much the score drops. "
                "A bigger drop means the feature matters more."
            )
            st.bar_chart(importance_table.set_index("feature")["importance"])
            st.dataframe(importance_table, width="stretch")

# ===== Tab 8: Predict =====
with tab_predict:
    st.subheader("Use the trained model")

    if "model_bytes" not in st.session_state:
        st.info("Run the final evaluation in the Evaluate tab first.")
    else:
        bundle = load_model_bundle(io.BytesIO(st.session_state["model_bytes"]))

        st.markdown("### 1. Download the trained model")
        st.caption(
            "One file with the preprocessing and the model together, so the same "
            "steps are applied to future data."
        )
        st.download_button(
            "⬇️ Download trained model (.joblib)",
            data=st.session_state["model_bytes"],
            file_name="trustml_model.joblib",
            mime="application/octet-stream",
        )

        st.divider()

        st.markdown("### 2. Predict on new data")
        st.caption(
            "The new file needs these columns: "
            + ", ".join(bundle["feature_columns"])
        )

        new_file = st.file_uploader(
            "Upload new data (CSV)", type=["csv"], key="predict_upload"
        )

        if new_file is not None:
            try:
                new_df = pd.read_csv(new_file)
                result_df, notes = predict_new_data(bundle, new_df)
            except ValueError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Could not make predictions: {error}")
            else:
                st.success(f"✅ Predicted {len(result_df):,} rows.")

                for note in notes:
                    st.info(note)

                st.dataframe(result_df, width="stretch")

                st.download_button(
                    "⬇️ Download predictions (CSV)",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

        st.divider()

        st.markdown("### 3. Use it in your own project")
        st.caption(
            "Download this script and keep it in the same folder as the model file. "
            "It prepares new data exactly like the training data and returns predictions."
        )

        script_text = generate_inference_script(bundle)

        st.download_button(
            "⬇️ Download predict.py",
            data=script_text.encode("utf-8"),
            file_name="predict.py",
            mime="text/x-python",
        )

        with st.expander("👀 Preview predict.py"):
            st.code(script_text, language="python")


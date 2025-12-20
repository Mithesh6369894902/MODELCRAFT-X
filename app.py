import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, r2_score

# -------------------- CONFIG -------------------- #
RANDOM_STATE = 42
st.set_page_config(
    page_title="ModelCraft-X",
    page_icon="🧠",
    layout="wide"
)

# -------------------- UI HEADER -------------------- #
st.title("🧠 ModelCraft-X")
st.subheader("Cross-Validated AutoML Benchmarking Framework")
st.caption("Pipeline-based | Explainable | Reproducible")

# -------------------- DATA UPLOAD -------------------- #
uploaded_file = st.file_uploader("📤 Upload CSV Dataset", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded Successfully")

    st.write("### 📊 Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

    # -------------------- TARGET SELECTION -------------------- #
    target_col = st.selectbox("🎯 Select Target Column", data.columns)

    X = data.drop(columns=[target_col])
    y = data[target_col]

    # -------------------- FEATURE TYPE DETECTION -------------------- #
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

    st.write("### 🔍 Automatic Feature Detection")
    col1, col2 = st.columns(2)
    col1.write("🔢 Numerical Features")
    col1.write(numerical_cols)
    col2.write("🔠 Categorical Features")
    col2.write(categorical_cols)

    # -------------------- TASK DETECTION -------------------- #
    if y.nunique() <= 10:
        task_type = "classification"
        scoring_metric = "accuracy"
        st.info("🧠 Detected Task Type: **CLASSIFICATION**")
    else:
        task_type = "regression"
        scoring_metric = "r2"
        st.info("🧠 Detected Task Type: **REGRESSION**")

    # -------------------- PREPROCESSING PIPELINE -------------------- #
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    # -------------------- MODEL ZOO -------------------- #
    def get_model_zoo(task):
        if task == "classification":
            return {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
                "Gradient Boosting": GradientBoostingClassifier()
            }
        else:
            return {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Random Forest Regressor": RandomForestRegressor(random_state=RANDOM_STATE)
            }

    model_zoo = get_model_zoo(task_type)

    # -------------------- BENCHMARKING ENGINE -------------------- #
    st.subheader("📊 Cross-Validated Model Benchmarking")

    benchmark_results = []

    with st.spinner("⏳ Running cross-validated benchmarking..."):
        for model_name, model in model_zoo.items():
            pipeline = Pipeline(steps=[
                ("preprocessing", preprocessor),
                ("model", model)
            ])

            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=5,
                scoring=scoring_metric
            )

            benchmark_results.append({
                "Model": model_name,
                "Mean CV Score": round(scores.mean(), 4),
                "Std Deviation": round(scores.std(), 4)
            })

    benchmark_df = pd.DataFrame(benchmark_results)
    benchmark_df = benchmark_df.sort_values(
        by="Mean CV Score",
        ascending=False
    ).reset_index(drop=True)

    st.dataframe(benchmark_df, use_container_width=True)

    # -------------------- BEST MODEL TRAINING -------------------- #
    st.subheader("🏆 Best Model Training")

    best_model_name = benchmark_df.iloc[0]["Model"]
    best_model = model_zoo[best_model_name]

    final_pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", best_model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    start_time = time.time()
    final_pipeline.fit(X_train, y_train)
    end_time = time.time()

    predictions = final_pipeline.predict(X_test)

    if task_type == "classification":
        final_score = accuracy_score(y_test, predictions)
        metric_label = "Accuracy"
    else:
        final_score = r2_score(y_test, predictions)
        metric_label = "R² Score"

    col1, col2 = st.columns(2)
    col1.metric(metric_label, round(final_score, 4))
    col2.metric("Training Time (sec)", round(end_time - start_time, 2))

    # -------------------- EXPERIMENT LOG -------------------- #
    st.subheader("🧾 Experiment Summary")

    experiment_log = {
        "Task Type": task_type,
        "Best Model": best_model_name,
        "Evaluation Metric": metric_label,
        "Final Score": round(final_score, 4),
        "Dataset Shape": data.shape,
        "Numerical Features": len(numerical_cols),
        "Categorical Features": len(categorical_cols),
        "Timestamp": time.ctime()
    }

    st.json(experiment_log)

else:
    st.info("👆 Upload a CSV file to start the AutoML benchmarking process")

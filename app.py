from __future__ import annotations

from pathlib import Path
import streamlit as st
import pandas as pd

from wastewater.dashboard_app import (
    discover_datasets,
    fit_model,
    load_dataset,
    plot_correlation,
    prepare_training_frame,
    summarise_dataset,
)

st.set_page_config(page_title="Respiratory Incidence Dashboard", page_icon="🫁", layout="wide")

st.title("Respiratory incidence predictive dashboard")
st.write("Upload or mount local datasets, inspect their structure, and build simple incidence forecasting models.")

root = Path.cwd()
available_datasets = discover_datasets(root)

with st.sidebar:
    st.header("Data source")
    selected_path = st.selectbox("Choose a dataset", [str(path) for path in available_datasets], index=0 if available_datasets else None)
    st.caption("The app scans the workspace for CSV, Parquet, and Excel files.")
    if st.button("Reload datasets"):
        st.rerun()

    uploaded_file = st.file_uploader("Or upload a CSV/Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    dataset_path = uploaded_file.name
    frame = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
else:
    if not selected_path:
        st.info("No datasets were found. Add data files to the workspace or upload one.")
        st.stop()
    dataset_path = selected_path
    frame = load_dataset(Path(dataset_path))

st.subheader("Dataset preview")
st.write(f"Loaded: {dataset_path}")
st.dataframe(frame.head(10), use_container_width=True)

numeric_columns = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
text_columns = [column for column in frame.columns if not pd.api.types.is_numeric_dtype(frame[column])]

col1, col2, col3 = st.columns(3)
col1.metric("Rows", len(frame))
col2.metric("Columns", len(frame.columns))
col3.metric("Numeric columns", len(numeric_columns))

if numeric_columns:
    st.subheader("Summary statistics")
    summary = summarise_dataset(frame)
    st.dataframe(summary, use_container_width=True)

    st.subheader("Correlation heatmap")
    fig = plot_correlation(frame, numeric_columns[:10])
    st.pyplot(fig)

    st.subheader("Model builder")
    target_column = st.selectbox("Target column", numeric_columns)
    feature_candidates = [column for column in numeric_columns if column != target_column]
    feature_columns = st.multiselect("Feature columns", feature_candidates, default=feature_candidates[:min(5, len(feature_candidates))])

    date_column = st.selectbox("Date column (optional)", [None] + [column for column in frame.columns if column.lower() in {"date", "datetime", "timestamp", "time"}], index=0)
    use_lags = st.checkbox("Use lagged features", value=True)
    lag_count = st.slider("Lag count", 1, 6, 3)
    model_name = st.selectbox("Model", ["linear", "ridge", "random_forest"])

    if feature_columns and st.button("Train model"):
        try:
            _, X, y = prepare_training_frame(
                frame,
                target_column,
                feature_columns,
                date_column=date_column,
                use_lags=use_lags,
                lag_count=lag_count,
            )
            result = fit_model(X, y, model_name=model_name)
            st.success("Model trained successfully")
            st.write("### Evaluation")
            st.write(result["metrics"])
            st.write("### Feature importance")
            st.dataframe(result["importances"].head(10), use_container_width=True)
        except Exception as exc:  # pragma: no cover - UI handling
            st.error(f"Model training failed: {exc}")
else:
    st.info("No numeric columns were detected. Select a dataset with numeric values for modeling.")

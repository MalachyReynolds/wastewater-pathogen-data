from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wastewater.dashboard.compat import make_streamlit_safe
from wastewater.dashboard.data import (
    build_custom_series,
    discover_local_files,
    list_series_catalogue,
    load_canonical_series,
    load_uploaded_or_local_file,
    merge_series,
    summarise_dataset,
)

st.set_page_config(page_title="Data", page_icon="\U0001f4c2", layout="wide")
st.title("Data")

st.header("Canonical series panel")
st.write(
    "The canonical panel combines every wastewater, clinical, UKHSA, Google Trends, "
    "OWID, and weather series the repository can discover into one long-format table. "
    "This is the data source used by the Explore, Model, and Forecast pages."
)

include_external = st.checkbox("Include external sources (OWID, weather)", value=True)
if st.button("Load canonical series panel"):
    with st.spinner("Loading series..."):
        series = load_canonical_series(ROOT, include_external=include_external)
    st.session_state["series"] = series
    st.session_state["catalogue"] = list_series_catalogue(series)

if "series" in st.session_state:
    series = st.session_state["series"]
    catalogue = st.session_state["catalogue"]
    st.success(f"Loaded {len(series):,} observations across {len(catalogue)} series.")
    st.dataframe(make_streamlit_safe(catalogue), width="stretch")
else:
    st.info("Click 'Load canonical series panel' to make data available to the other pages.")

st.divider()
st.header("Preview a single raw file")
st.write("Browse or upload an individual dataset file to preview its raw structure.")

root = Path.cwd()
available_files = discover_local_files(root)

with st.sidebar:
    st.header("Raw file preview")
    selected_path = st.selectbox(
        "Choose a file",
        [str(path) for path in available_files],
        index=0 if available_files else None,
    )
    if st.button("Reload file list"):
        st.rerun()
    uploaded_file = st.file_uploader("Or upload a CSV/Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    raw_frame = load_uploaded_or_local_file(uploaded_file=uploaded_file)
    raw_label = uploaded_file.name
elif selected_path:
    raw_frame = load_uploaded_or_local_file(path=Path(selected_path))
    raw_label = selected_path
else:
    raw_frame = None
    raw_label = None

if raw_frame is not None:
    st.session_state["raw_frame"] = raw_frame
    st.session_state["raw_label"] = raw_label

    st.write(f"Loaded: {raw_label}")
    st.dataframe(make_streamlit_safe(raw_frame.head(10)), width="stretch")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(raw_frame))
    col2.metric("Columns", len(raw_frame.columns))
    col3.metric("Numeric columns", sum(raw_frame.dtypes.apply(lambda dtype: dtype.kind in "iuf")))

    st.subheader("Summary statistics")
    st.dataframe(make_streamlit_safe(summarise_dataset(raw_frame)), width="stretch")

    st.subheader("Add to series panel for modelling")
    st.write(
        "Register this file as an extra predictor or target series so it can be used "
        "alongside the canonical panel on the Explore, Model, and Forecast pages."
    )

    numeric_columns = [column for column in raw_frame.columns if pd.api.types.is_numeric_dtype(raw_frame[column])]
    if not numeric_columns:
        st.info("This file has no numeric columns, so it cannot be added as a series.")
    else:
        col1, col2, col3 = st.columns(3)
        date_column = col1.selectbox("Date column", raw_frame.columns.tolist(), key="custom_date_column")
        value_column = col2.selectbox("Value column", numeric_columns, key="custom_value_column")
        role = col3.radio("Role", ["predictive", "predicted"], key="custom_role")

        default_name = Path(raw_label).stem if raw_label else "custom_series"
        series_name = st.text_input("Series name", value=default_name, key="custom_series_name")

        if st.button("Add to series panel"):
            try:
                series_id = f"custom::{series_name.strip().lower().replace(' ', '_')}"
                custom_series = build_custom_series(
                    raw_frame,
                    date_column=date_column,
                    value_column=value_column,
                    series_id=series_id,
                    series_name=series_name,
                    role=role,
                    source_file=raw_label,
                )
                merged = merge_series(st.session_state.get("series"), custom_series)
                st.session_state["series"] = merged
                st.session_state["catalogue"] = list_series_catalogue(merged)
                st.success(f"Added '{series_id}' with {len(custom_series):,} rows to the series panel.")
            except Exception as exc:
                st.error(f"Could not add series: {exc}")
else:
    st.info("No file selected. Add data files to the workspace or upload one.")

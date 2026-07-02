from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wastewater.dashboard.agent_data import (
    feature_table_to_canonical_series,
    list_latest_agent_manifests,
    load_feature_table,
    load_normalized_signal_tables,
    normalized_signals_to_canonical_series,
)
from wastewater.dashboard.data import list_series_catalogue, merge_series

st.set_page_config(page_title="Agent Data", page_icon="🤖", layout="wide")
st.title("Agent Data")
st.write(
    "Load versioned Parquet outputs produced by the autonomous respiratory data agent. "
    "The loaded artifacts are adapted into the dashboard's canonical series panel, so the "
    "Explore, Model, and Forecast pages can use them directly."
)

st.info(
    "Expected layout: `data_registry/latest/*.json` points to feature-table manifests, "
    "and manifests point to Parquet files under `data/features/`. Normalized signal tables "
    "can also be loaded from `data/normalized/**/*.parquet`."
)


@st.cache_data(show_spinner=False)
def cached_latest_manifests(root: str) -> pd.DataFrame:
    return list_latest_agent_manifests(Path(root))


@st.cache_data(show_spinner=False)
def cached_feature_table(root: str, feature_set: str) -> tuple[pd.DataFrame, dict]:
    return load_feature_table(Path(root), feature_set)


@st.cache_data(show_spinner=False)
def cached_normalized_signals(root: str) -> pd.DataFrame:
    return load_normalized_signal_tables(Path(root))


st.header("Latest registered feature tables")
manifest_table = cached_latest_manifests(str(ROOT))

if manifest_table.empty:
    st.warning(
        "No latest agent manifests were found yet. Run the agent storage pipeline first, "
        "or place a latest pointer under `data_registry/latest/`."
    )
else:
    st.dataframe(manifest_table, width="stretch")
    feature_sets = manifest_table["feature_set"].dropna().astype(str).tolist()
    selected_feature_set = st.selectbox("Feature table", feature_sets)

    if st.button("Load selected feature table"):
        try:
            feature_frame, manifest = cached_feature_table(str(ROOT), selected_feature_set)
            st.session_state["agent_feature_frame"] = feature_frame
            st.session_state["agent_feature_manifest"] = manifest
            st.success(f"Loaded {selected_feature_set}: {len(feature_frame):,} rows, {len(feature_frame.columns):,} columns.")
        except Exception as exc:
            st.error(f"Could not load feature table: {exc}")

if "agent_feature_frame" in st.session_state:
    feature_frame = st.session_state["agent_feature_frame"]
    manifest = st.session_state["agent_feature_manifest"]
    feature_set = str(manifest.get("feature_set", "agent_feature_table"))
    source_file = str(manifest.get("path", "data/features"))

    st.subheader("Feature table preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(feature_frame):,}")
    col2.metric("Columns", f"{len(feature_frame.columns):,}")
    col3.metric("Manifest", manifest.get("run_id", "latest"))
    st.dataframe(feature_frame.head(100), width="stretch")

    if st.button("Add feature table to modelling panel"):
        try:
            agent_series = feature_table_to_canonical_series(
                feature_frame,
                feature_set=feature_set,
                source_file=source_file,
            )
            if agent_series.empty:
                st.warning("No usable numeric feature series were found in this table.")
            else:
                merged = merge_series(st.session_state.get("series"), agent_series)
                st.session_state["series"] = merged
                st.session_state["catalogue"] = list_series_catalogue(merged)
                st.success(f"Added {agent_series['series_id'].nunique():,} agent feature series to the modelling panel.")
        except Exception as exc:
            st.error(f"Could not convert feature table into dashboard series: {exc}")

st.divider()
st.header("Normalized agent signals")
st.write(
    "This loads long-form normalized signal tables from `data/normalized/**/*.parquet`, "
    "for example UKHSA, wastewater, weather, or search-trend signals after the agent has "
    "mapped them into the shared respiratory schema."
)

if st.button("Load normalized agent signals"):
    try:
        normalized = cached_normalized_signals(str(ROOT))
        st.session_state["agent_normalized_signals"] = normalized
        if normalized.empty:
            st.warning("No normalized Parquet signal tables were found under `data/normalized/`.")
        else:
            st.success(f"Loaded {len(normalized):,} normalized signal rows.")
    except Exception as exc:
        st.error(f"Could not load normalized signals: {exc}")

if "agent_normalized_signals" in st.session_state and not st.session_state["agent_normalized_signals"].empty:
    normalized = st.session_state["agent_normalized_signals"]
    st.subheader("Normalized signal preview")
    st.dataframe(normalized.head(100), width="stretch")

    if st.button("Add normalized signals to modelling panel"):
        try:
            signal_series = normalized_signals_to_canonical_series(normalized)
            if signal_series.empty:
                st.warning("No usable normalized signal series were found.")
            else:
                merged = merge_series(st.session_state.get("series"), signal_series)
                st.session_state["series"] = merged
                st.session_state["catalogue"] = list_series_catalogue(merged)
                st.success(f"Added {signal_series['series_id'].nunique():,} normalized signal series to the modelling panel.")
        except Exception as exc:
            st.error(f"Could not convert normalized signals into dashboard series: {exc}")

st.divider()
st.header("Current modelling panel")
if "series" in st.session_state:
    catalogue = st.session_state.get("catalogue") or list_series_catalogue(st.session_state["series"])
    st.success(f"The modelling panel contains {len(st.session_state['series']):,} observations across {len(catalogue):,} series.")
    st.dataframe(catalogue, width="stretch")
else:
    st.info("No modelling panel is loaded yet. Load canonical data on the Data page or add agent artifacts above.")

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wastewater.dashboard.charts import correlation_heatmap, timeseries_chart
from wastewater.ml_panel import PanelBuildConfig, build_wide_panel

st.set_page_config(page_title="Explore", page_icon="\U0001f4c8", layout="wide")
st.title("Explore")

if "series" not in st.session_state:
    st.warning("Load the canonical series panel on the Data page first.")
    st.stop()

series = st.session_state["series"]
catalogue = st.session_state["catalogue"]

st.header("Time series")
series_options = catalogue["series_id"].tolist()
default_selection = series_options[: min(5, len(series_options))]
selected_series = st.multiselect("Series to plot", series_options, default=default_selection)

if selected_series:
    filtered = series[series["series_id"].isin(selected_series)]
    fig = timeseries_chart(filtered, x="date", y="value", color="series_id", title="Selected series over time")
    st.plotly_chart(fig)
else:
    st.info("Select at least one series to plot.")

st.divider()
st.header("Correlation heatmap")
st.write("Correlation is computed on a weekly-aggregated wide panel of the selected series.")

correlation_series = st.multiselect(
    "Series to correlate",
    series_options,
    default=series_options[: min(10, len(series_options))],
    key="correlation_series",
)

if len(correlation_series) >= 2:
    subset = series[series["series_id"].isin(correlation_series)]
    wide = build_wide_panel(subset, config=PanelBuildConfig())
    numeric_columns = [column for column in wide.columns if column != "period"]
    fig = correlation_heatmap(wide, numeric_columns)
    st.plotly_chart(fig)
else:
    st.info("Select at least two series to compute a correlation heatmap.")

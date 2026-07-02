from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wastewater.dashboard.charts import forecast_fan_chart
from wastewater.dashboard.forecasting import forecast_target
from wastewater.ml_panel import PanelBuildConfig

st.set_page_config(page_title="Forecast", page_icon="\U0001f52e", layout="wide")
st.title("Forecast")
st.write(
    "Project a target series past its last observed date, using either a leading indicator "
    "whose data already extends further forward, or an autoregressive projection of the "
    "target's own history when no such indicator is available."
)

if "series" not in st.session_state:
    st.warning("Load the canonical series panel on the Data page first.")
    st.stop()

series = st.session_state["series"]
catalogue = st.session_state["catalogue"]
predicted_ids = catalogue.loc[catalogue["role"] == "predicted", "series_id"].tolist()

if not predicted_ids:
    st.warning("No series are marked as a predicted target in the loaded data.")
    st.stop()

target_id = st.selectbox("Target series", predicted_ids)
target_date_max = catalogue.loc[catalogue["series_id"] == target_id, "date_max"].iloc[0]

leading_indicator_options = catalogue.loc[
    (catalogue["role"] == "predictive") & (catalogue["date_max"] > target_date_max), "series_id"
].tolist()

col1, col2, col3 = st.columns(3)
horizon = col1.slider("Forecast horizon (periods)", 1, 12, 4)
freq = "M" if col2.radio("Frequency", ["Weekly", "Monthly"]) == "Monthly" else "W"
confidence = col3.slider("Confidence interval (%)", 50, 95, 80)

predictor_id = st.selectbox(
    "Leading indicator predictor",
    ["Autoregressive (target's own history)"] + leading_indicator_options,
)
predictor_id = None if predictor_id == "Autoregressive (target's own history)" else predictor_id

if st.button("Run forecast"):
    config = PanelBuildConfig(freq=freq)
    try:
        predictions, meta = forecast_target(
            series,
            target_id=target_id,
            horizon_periods=horizon,
            config=config,
            predictor_id=predictor_id,
            interval_level=confidence / 100.0,
        )
        st.session_state["last_forecast_predictions"] = predictions
        st.session_state["last_forecast_meta"] = meta
    except Exception as exc:
        st.error(f"Forecasting failed: {exc}")

if "last_forecast_predictions" in st.session_state:
    predictions = st.session_state["last_forecast_predictions"]
    meta = st.session_state["last_forecast_meta"]

    if meta["strategy"] == "leading_indicator":
        st.success(f"Using {meta['predictor_id']} as a leading indicator.")
    else:
        message = "Using an autoregressive projection of the target's own history."
        if meta.get("requested_predictor_id"):
            message += f" ({meta['requested_predictor_id']} could not be used: {meta['fallback_reason']})"
        st.info(message)

    fig = forecast_fan_chart(predictions)
    st.plotly_chart(fig)
    st.dataframe(predictions[predictions["is_forecast"]], width="stretch")

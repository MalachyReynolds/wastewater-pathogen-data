from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wastewater.dashboard.charts import actual_vs_predicted_chart
from wastewater.dashboard.models import MODEL_REGISTRY, fit_models_for_targets, fit_spike_model
from wastewater.ml_panel import PanelBuildConfig
from wastewater.spike_neural_network import SpikeNNConfig

st.set_page_config(page_title="Model", page_icon="\U0001f9ea", layout="wide")
st.title("Model builder")

if "series" not in st.session_state:
    st.warning("Load the canonical series panel on the Data page first.")
    st.stop()

series = st.session_state["series"]
catalogue = st.session_state["catalogue"]
predicted_ids = catalogue.loc[catalogue["role"] == "predicted", "series_id"].tolist()
predictive_ids = catalogue.loc[catalogue["role"] == "predictive", "series_id"].tolist()

if not predicted_ids:
    st.warning("No series are marked as a predicted target in the loaded data.")
    st.stop()

st.header("Regression models")
st.write("Predictors and targets can each be drawn from multiple datasets in the loaded panel.")

target_ids = st.multiselect(
    "Target series (one model is fit per target)", predicted_ids, default=predicted_ids[: min(1, len(predicted_ids))]
)
predictor_ids = st.multiselect("Predictor series", predictive_ids, default=predictive_ids)
model_labels = st.multiselect("Models to fit", list(MODEL_REGISTRY.keys()), default=list(MODEL_REGISTRY.keys()))
col1, col2 = st.columns(2)
freq = "M" if col1.radio("Frequency", ["Weekly", "Monthly"]) == "Monthly" else "W"
lag_count = col2.slider("Lags to use as features", 1, 8, 4)

if st.button("Fit models"):
    if not target_ids:
        st.error("Select at least one target series.")
    elif not predictor_ids:
        st.error("Select at least one predictor series.")
    else:
        model_names = [MODEL_REGISTRY[label] for label in model_labels]
        config = PanelBuildConfig(freq=freq, lags=tuple(range(1, lag_count + 1)))
        try:
            metrics, predictions = fit_models_for_targets(
                series, target_ids=target_ids, config=config, model_names=model_names, predictor_ids=predictor_ids
            )
            if metrics.empty:
                st.error(
                    "No model could be fit. Check that the targets have enough selected predictors with lagged coverage."
                )
            else:
                st.session_state["last_model_metrics"] = metrics
                st.session_state["last_model_predictions"] = predictions
        except Exception as exc:
            st.error(f"Model fitting failed: {exc}")

if "last_model_metrics" in st.session_state:
    metrics = st.session_state["last_model_metrics"]
    predictions = st.session_state["last_model_predictions"]
    st.subheader("Held-out metrics")
    st.dataframe(metrics, width="stretch")

    fitted_targets = sorted(predictions["target_id"].unique()) if not predictions.empty else []
    if fitted_targets:
        col1, col2 = st.columns(2)
        chosen_target = col1.selectbox("Target", fitted_targets)
        target_predictions = predictions[predictions["target_id"] == chosen_target]
        fit_models_available = sorted(target_predictions["model"].unique())
        chosen_model = col2.selectbox("Model", fit_models_available)
        model_predictions = target_predictions[target_predictions["model"] == chosen_model]
        fig = actual_vs_predicted_chart(
            model_predictions["target"], model_predictions["prediction"], dates=model_predictions["period"]
        )
        st.plotly_chart(fig)

st.divider()
st.header("Neural network: spike early-warning")
st.write(
    "This model classifies the probability of a future spike and estimates its severity with a "
    "bootstrap confidence interval, rather than a single continuous prediction, so it is shown separately."
)

spike_target_id = st.selectbox("Target series for spike detection", predicted_ids, key="spike_target")
horizons = st.multiselect("Forecast horizons (periods ahead)", [1, 2, 3, 4, 6, 8], default=[1, 2, 3, 4])

if st.button("Fit spike early-warning model"):
    if not horizons:
        st.error("Select at least one horizon.")
    else:
        config = SpikeNNConfig(horizons=tuple(sorted(horizons)))
        try:
            results, predictions = fit_spike_model(series, target_id=spike_target_id, config=config)
            if results.empty:
                st.error("No spike model could be fit for this target.")
            else:
                st.session_state["last_spike_results"] = results
                st.session_state["last_spike_predictions"] = predictions
        except Exception as exc:
            st.error(f"Spike model fitting failed: {exc}")

if "last_spike_results" in st.session_state:
    st.subheader("Spike-classification metrics by horizon")
    st.dataframe(st.session_state["last_spike_results"], width="stretch")
    st.subheader("Severity predictions")
    st.dataframe(st.session_state["last_spike_predictions"], width="stretch")

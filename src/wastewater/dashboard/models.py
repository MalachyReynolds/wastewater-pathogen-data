from __future__ import annotations

import pandas as pd

from ..ml_panel import PanelBuildConfig, build_lagged_feature_panel, evaluate_models_for_target
from ..spike_neural_network import SpikeNNConfig, fit_spike_nn_for_target

MODEL_REGISTRY: dict[str, str] = {
    "Ordinary least squares": "ols",
    "Ridge (cross-validated)": "ridge",
    "Elastic net (cross-validated)": "elastic_net",
    "Random forest": "random_forest",
    "Histogram gradient boosting": "hist_gradient_boosting",
}

SPIKE_MODEL_LABEL = "Neural network (spike early-warning)"


def fit_models(
    series: pd.DataFrame,
    target_id: str,
    config: PanelBuildConfig = PanelBuildConfig(),
    model_names: list[str] | None = None,
    predictor_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the regression model registry for one target, optionally filtered to a subset.

    ``predictor_ids``, when given, restricts the predictive series used as lagged
    features to that subset instead of every predictive series in ``series`` --
    filtering the input frame rather than changing ``ml_panel.build_lagged_feature_panel``,
    which otherwise uses every predictive series it finds.
    """
    working = series
    if predictor_ids is not None:
        is_target = working["series_id"] == target_id
        is_chosen_predictor = (working["role"] == "predictive") & working["series_id"].isin(predictor_ids)
        working = working[is_target | is_chosen_predictor]

    panel, feature_cols = build_lagged_feature_panel(working, target_id=target_id, config=config)
    metrics, predictions = evaluate_models_for_target(panel, feature_cols, target_id=target_id)

    if model_names is None or metrics.empty:
        return metrics, predictions

    metrics = metrics[metrics["model"].isin(model_names)].reset_index(drop=True)
    predictions = predictions[predictions["model"].isin(model_names)].reset_index(drop=True)
    return metrics, predictions


def fit_models_for_targets(
    series: pd.DataFrame,
    target_ids: list[str],
    config: PanelBuildConfig = PanelBuildConfig(),
    model_names: list[str] | None = None,
    predictor_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the regression model registry independently for each of several targets."""
    metrics_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    for target_id in target_ids:
        metrics, predictions = fit_models(
            series, target_id=target_id, config=config, model_names=model_names, predictor_ids=predictor_ids
        )
        if not metrics.empty:
            metrics_parts.append(metrics)
        if not predictions.empty:
            prediction_parts.append(predictions)

    combined_metrics = pd.concat(metrics_parts, ignore_index=True) if metrics_parts else pd.DataFrame()
    combined_predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    return combined_metrics, combined_predictions


def fit_spike_model(
    series: pd.DataFrame,
    target_id: str,
    config: SpikeNNConfig = SpikeNNConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the spike early-warning neural network for one target."""
    return fit_spike_nn_for_target(series, target_id=target_id, config=config)

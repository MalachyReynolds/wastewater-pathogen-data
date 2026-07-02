from __future__ import annotations

import numpy as np
import pandas as pd

from wastewater.dashboard.models import MODEL_REGISTRY, fit_models, fit_models_for_targets, fit_spike_model
from wastewater.ml_panel import model_specs
from wastewater.spike_neural_network import SpikeNNConfig


def _synthetic_series(n_periods: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n_periods, freq="W")
    rng = np.random.default_rng(0)
    predictor = np.linspace(0, 10, n_periods) + rng.normal(scale=0.1, size=n_periods)
    target = np.roll(predictor, 1) * 2 + rng.normal(scale=0.1, size=n_periods)

    predictor_df = pd.DataFrame(
        {
            "date": dates,
            "value": predictor,
            "series_id": "predictor_a",
            "role": "predictive",
            "dataset_family": "fam_a",
            "series_name": "Predictor A",
            "source_file": "a.csv",
        }
    )
    target_df = pd.DataFrame(
        {
            "date": dates,
            "value": target,
            "series_id": "target_a",
            "role": "predicted",
            "dataset_family": "fam_b",
            "series_name": "Target A",
            "source_file": "b.csv",
        }
    )
    return pd.concat([predictor_df, target_df], ignore_index=True)


def _synthetic_series_multi(n_periods: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n_periods, freq="W")
    rng = np.random.default_rng(0)
    predictor_a = np.linspace(0, 10, n_periods) + rng.normal(scale=0.1, size=n_periods)
    predictor_b = np.linspace(5, 15, n_periods) + rng.normal(scale=0.1, size=n_periods)
    target_a = np.roll(predictor_a, 1) * 2 + rng.normal(scale=0.1, size=n_periods)
    target_b = np.roll(predictor_b, 1) * 3 + rng.normal(scale=0.1, size=n_periods)

    def _frame(values, series_id, role, family):
        return pd.DataFrame(
            {
                "date": dates,
                "value": values,
                "series_id": series_id,
                "role": role,
                "dataset_family": family,
                "series_name": series_id,
                "source_file": f"{series_id}.csv",
            }
        )

    return pd.concat(
        [
            _frame(predictor_a, "predictor_a", "predictive", "fam_p"),
            _frame(predictor_b, "predictor_b", "predictive", "fam_p"),
            _frame(target_a, "target_a", "predicted", "fam_t"),
            _frame(target_b, "target_b", "predicted", "fam_t"),
        ],
        ignore_index=True,
    )


def test_model_registry_matches_ml_panel_specs():
    assert set(MODEL_REGISTRY.values()) == set(model_specs().keys())


def test_fit_models_filters_to_requested_subset():
    series = _synthetic_series()
    metrics, predictions = fit_models(series, target_id="target_a", model_names=["ridge", "ols"])

    assert not metrics.empty
    assert set(metrics["model"]) <= {"ridge", "ols"}
    assert set(predictions["model"]) <= {"ridge", "ols"}


def test_fit_models_predictor_ids_restricts_features():
    series = _synthetic_series_multi()

    metrics_both, _ = fit_models(series, target_id="target_a", model_names=["ridge"])
    metrics_one, _ = fit_models(series, target_id="target_a", model_names=["ridge"], predictor_ids=["predictor_a"])

    assert not metrics_both.empty and not metrics_one.empty
    assert metrics_one["n_features"].iloc[0] < metrics_both["n_features"].iloc[0]


def test_fit_models_for_targets_batches_across_multiple_targets():
    series = _synthetic_series_multi()
    metrics, predictions = fit_models_for_targets(
        series, target_ids=["target_a", "target_b"], model_names=["ridge", "ols"], predictor_ids=["predictor_a", "predictor_b"]
    )

    assert set(metrics["target_id"]) == {"target_a", "target_b"}
    assert set(predictions["target_id"]) == {"target_a", "target_b"}


def test_fit_spike_model_runs_end_to_end():
    series = _synthetic_series(n_periods=60)
    config = SpikeNNConfig(horizons=(1, 2), n_bootstrap_models=3, max_iter=200)
    results, predictions = fit_spike_model(series, target_id="target_a", config=config)

    assert not results.empty
    assert set(results["horizon"]) <= {1, 2}

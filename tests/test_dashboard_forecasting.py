from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wastewater.dashboard.forecasting import forecast_autoregressive, forecast_target
from wastewater.ml_panel import PanelBuildConfig


def _autoregressive_series(n_periods: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=n_periods, freq="W-MON")
    rng = np.random.default_rng(1)
    trend = np.linspace(5, 15, n_periods)
    values = trend + rng.normal(scale=0.05, size=n_periods)
    return pd.DataFrame(
        {
            "date": dates,
            "value": values,
            "series_id": "target_a",
            "role": "predicted",
            "dataset_family": "fam_a",
            "series_name": "Target A",
            "source_file": "a.csv",
        }
    )


def _leading_indicator_series(n_periods: int = 60, predictor_lead: int = 8) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=n_periods, freq="W-MON")
    predictor_dates = pd.date_range("2023-01-02", periods=n_periods + predictor_lead, freq="W-MON")
    rng = np.random.default_rng(2)

    predictor_values = np.linspace(0, 20, len(predictor_dates)) + rng.normal(scale=0.05, size=len(predictor_dates))
    predictor_df = pd.DataFrame(
        {
            "date": predictor_dates,
            "value": predictor_values,
            "series_id": "predictor_a",
            "role": "predictive",
            "dataset_family": "fam_p",
            "series_name": "Predictor A",
            "source_file": "p.csv",
        }
    )

    lag = 2
    target_values = np.concatenate([np.full(lag, predictor_values[0]), predictor_values[: n_periods - lag]]) * 2.0
    target_values = target_values[:n_periods] + rng.normal(scale=0.05, size=n_periods)
    target_df = pd.DataFrame(
        {
            "date": dates,
            "value": target_values,
            "series_id": "target_a",
            "role": "predicted",
            "dataset_family": "fam_a",
            "series_name": "Target A",
            "source_file": "a.csv",
        }
    )
    return pd.concat([predictor_df, target_df], ignore_index=True)


def test_forecast_autoregressive_produces_horizon_rows_with_valid_intervals():
    series = _autoregressive_series()
    predictions, meta = forecast_autoregressive(
        series, target_id="target_a", horizon_periods=5, config=PanelBuildConfig(), n_bootstrap=5
    )

    forecast_rows = predictions[predictions["is_forecast"]]
    assert len(forecast_rows) == 5
    assert forecast_rows["period"].is_monotonic_increasing
    assert (forecast_rows["lower"] <= forecast_rows["prediction"]).all()
    assert (forecast_rows["prediction"] <= forecast_rows["upper"]).all()
    assert forecast_rows["actual"].isna().all()
    assert meta["strategy"] == "autoregressive"


def test_forecast_autoregressive_rejects_non_predicted_target():
    series = _autoregressive_series()
    series = series.assign(role="predictive")
    with pytest.raises(ValueError):
        forecast_autoregressive(series, target_id="target_a", horizon_periods=3)


def test_forecast_target_uses_leading_indicator_when_available():
    series = _leading_indicator_series()
    predictions, meta = forecast_target(
        series,
        target_id="target_a",
        horizon_periods=4,
        predictor_id="predictor_a",
        n_bootstrap=5,
    )

    forecast_rows = predictions[predictions["is_forecast"]]
    assert len(forecast_rows) == 4
    assert meta["strategy"] == "leading_indicator"
    assert (forecast_rows["lower"] <= forecast_rows["upper"]).all()


def test_forecast_target_falls_back_to_autoregressive_without_future_predictor_rows():
    series = _autoregressive_series()
    flat_predictor = series.copy()
    flat_predictor["series_id"] = "predictor_flat"
    flat_predictor["role"] = "predictive"
    combined = pd.concat([series, flat_predictor], ignore_index=True)

    predictions, meta = forecast_target(
        combined,
        target_id="target_a",
        horizon_periods=3,
        predictor_id="predictor_flat",
        n_bootstrap=5,
    )

    assert meta["strategy"] == "autoregressive"
    assert meta["requested_predictor_id"] == "predictor_flat"
    assert (predictions[predictions["is_forecast"]]["prediction"].notna()).all()

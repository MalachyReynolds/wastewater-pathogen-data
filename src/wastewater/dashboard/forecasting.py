"""Forward forecasting for respiratory-incidence series, past the last observed date.

Every other model in ``ml_panel``/``spike_neural_network``/``leakage_safe_matrix``
evaluates on a historical train/test split; none projects a target beyond its
last observation. This module adds two strategies for genuine forward
projection:

  * a leading-indicator forecast, when a predictor series has data that already
    extends past the target's last observed date (``rolling_origin_pair_forecast``
    reused for the historical backtest, plus one extra fit to project forward);
  * an autoregressive recursive forecast on the target's own lagged history,
    with bootstrap confidence intervals, for targets with no such predictor.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.linear_model import Ridge

from ..leakage_safe_matrix import aggregate_one_series, normalise_using_train, rolling_origin_pair_forecast
from ..ml_panel import PanelBuildConfig, build_wide_panel, model_specs

PREDICTION_COLUMNS = ["period", "actual", "prediction", "lower", "upper", "is_forecast"]


def _future_period_index(last_period, horizon: int, freq: str) -> pd.DatetimeIndex:
    step_freq = "MS" if freq.upper().startswith("M") else "7D"
    return pd.date_range(start=pd.Timestamp(last_period), periods=horizon + 1, freq=step_freq)[1:]


def forecast_with_leading_indicator(
    series: pd.DataFrame,
    predictor_id: str,
    target_id: str,
    horizon_periods: int,
    *,
    freq: str = "W",
    lags: Iterable[int] = (1, 2, 3, 4),
    aggregation: str = "mean",
    ridge_alpha: float = 1.0,
    interval_level: float = 0.80,
    initial_train_fraction: float = 0.6,
    min_train_size: int = 12,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Forecast a target past its last observation using a leading-indicator predictor.

    Reuses ``rolling_origin_pair_forecast`` for the historical backtest, then fits
    one additional model on the full paired history and applies it to predictor
    rows dated after the target's last observation.
    """
    if horizon_periods <= 0:
        raise ValueError("horizon_periods must be positive")

    predictor = series[series["series_id"] == predictor_id]
    target = series[series["series_id"] == target_id]
    if predictor.empty or target.empty:
        raise ValueError("Both predictor_id and target_id must exist in the series panel.")

    backtest_result, backtest_predictions = rolling_origin_pair_forecast(
        predictor,
        target,
        predictor_id=predictor_id,
        target_id=target_id,
        freq=freq,
        lags=lags,
        aggregation=aggregation,
        initial_train_fraction=initial_train_fraction,
        min_train_size=min_train_size,
        ridge_alpha=ridge_alpha,
    )

    clean_lags = sorted({int(lag) for lag in lags})
    predictor_cols = [f"x_lag{lag}" for lag in clean_lags]

    x = aggregate_one_series(predictor, freq=freq, aggregation=aggregation).rename(columns={"value": "x"})
    y = aggregate_one_series(target, freq=freq, aggregation=aggregation).rename(columns={"value": "y"})
    for lag in clean_lags:
        x[f"x_lag{lag}"] = x["x"].shift(lag)

    paired = pd.merge(x, y, on="period", how="inner").dropna(subset=[*predictor_cols, "y"]).sort_values("period")
    if len(paired) < min_train_size + 1:
        raise ValueError("Not enough paired history to fit a leading-indicator forecast model.")

    X_all_raw = paired[predictor_cols].astype(float)
    y_all_raw = paired["y"].astype(float)
    X_all_z, x_mean, x_std = normalise_using_train(X_all_raw, X_all_raw)
    y_all_z, y_mean, y_std = normalise_using_train(y_all_raw, y_all_raw)

    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model.fit(X_all_z, y_all_z)

    fitted = model.predict(X_all_z) * y_std + y_mean
    residual_sigma = float(np.std(y_all_raw.to_numpy() - fitted, ddof=1)) if len(paired) > 2 else float(y_all_raw.std())
    if not np.isfinite(residual_sigma) or residual_sigma <= 0:
        residual_sigma = float(y_all_raw.std()) if y_all_raw.std() > 0 else 1.0

    target_last_period = y["period"].max()
    future_rows = (
        x[(x["period"] > target_last_period) & x[predictor_cols].notna().all(axis=1)]
        .sort_values("period")
        .head(horizon_periods)
    )
    if future_rows.empty:
        raise ValueError("The selected predictor has no rows dated after the target's last observation.")

    X_future_z = (future_rows[predictor_cols].astype(float) - x_mean) / x_std
    future_pred = model.predict(X_future_z) * y_std + y_mean

    z_value = float(norm.ppf(0.5 + interval_level / 2.0))
    lower = future_pred - z_value * residual_sigma
    upper = future_pred + z_value * residual_sigma

    historical = pd.DataFrame(
        {
            "period": backtest_predictions["period"],
            "actual": backtest_predictions["actual"],
            "prediction": backtest_predictions["prediction"],
            "lower": np.nan,
            "upper": np.nan,
            "is_forecast": False,
        }
    )
    future = pd.DataFrame(
        {
            "period": future_rows["period"].to_numpy(),
            "actual": np.nan,
            "prediction": future_pred,
            "lower": lower,
            "upper": upper,
            "is_forecast": True,
        }
    )
    predictions = pd.concat([historical, future], ignore_index=True)[PREDICTION_COLUMNS]

    meta = {
        "strategy": "leading_indicator",
        "predictor_id": predictor_id,
        "target_id": target_id,
        "model": "ridge",
        "n_forecast_periods": len(future),
        "interval_level": interval_level,
        "backtest_metrics": backtest_result,
    }
    return predictions, meta


def _build_autoregressive_panel(
    wide: pd.DataFrame, target_id: str, lags: Iterable[int]
) -> tuple[pd.DataFrame, list[str], list[int]]:
    """Build a supervised panel for one target using only its own lagged history."""
    clean_lags = sorted({int(lag) for lag in lags})
    panel = wide[["period", target_id]].copy().rename(columns={target_id: "target"})
    feature_cols: list[str] = []
    for lag in clean_lags:
        col = f"target_lag{lag}"
        panel[col] = panel["target"].shift(lag)
        feature_cols.append(col)
    panel = panel.dropna(subset=["target", *feature_cols]).sort_values("period").reset_index(drop=True)
    return panel, feature_cols, clean_lags


def _recursive_forecast(
    model, feature_cols: list[str], lags: list[int], history_values: list[float], horizon: int
) -> list[float]:
    """Project ``horizon`` steps ahead, feeding each new prediction back in as a lag."""
    values = list(history_values)
    max_lag = max(lags)
    predictions: list[float] = []
    for _ in range(horizon):
        window = values[-max_lag:]
        row = {col: window[-lag] for col, lag in zip(feature_cols, lags)}
        X = pd.DataFrame([row])[feature_cols]
        prediction = float(model.predict(X)[0])
        predictions.append(prediction)
        values.append(prediction)
    return predictions


def forecast_autoregressive(
    series: pd.DataFrame,
    target_id: str,
    horizon_periods: int,
    config: PanelBuildConfig = PanelBuildConfig(),
    *,
    interval_level: float = 0.80,
    n_bootstrap: int = 25,
    random_state: int = 42,
    min_train_size: int = 12,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Recursively forecast a target using only its own lagged history.

    Confidence bands come from a bootstrap ensemble of resampled Ridge fits,
    following the same resample-and-quantile pattern used for severity
    intervals in ``spike_neural_network._bootstrap_severity_interval``.
    """
    if horizon_periods <= 0:
        raise ValueError("horizon_periods must be positive")

    predicted_ids = set(series.loc[series["role"] == "predicted", "series_id"].dropna().unique())
    if target_id not in predicted_ids:
        raise ValueError(f"Target is not a predicted series: {target_id}")

    wide = build_wide_panel(series, config=config)
    if wide.empty or target_id not in wide.columns:
        raise ValueError(f"No data available for target: {target_id}")

    panel, feature_cols, clean_lags = _build_autoregressive_panel(wide, target_id, config.lags)
    if len(panel) < min_train_size:
        raise ValueError(f"Not enough historical rows to fit an autoregressive forecast: {len(panel)}")

    history = wide[["period", target_id]].dropna().sort_values("period")
    history_values = history[target_id].astype(float).tolist()
    future_periods = _future_period_index(history["period"].max(), horizon_periods, config.freq)

    base_pipeline = model_specs(random_state=random_state)["ridge"]
    rng = np.random.default_rng(random_state)
    bootstrap_paths: list[list[float]] = []
    for _ in range(n_bootstrap):
        sample_index = rng.integers(0, len(panel), size=len(panel))
        model = clone(base_pipeline)
        try:
            model.fit(panel.iloc[sample_index][feature_cols], panel.iloc[sample_index]["target"])
            bootstrap_paths.append(_recursive_forecast(model, feature_cols, clean_lags, history_values, horizon_periods))
        except Exception:
            continue

    if not bootstrap_paths:
        raise ValueError("Bootstrap forecasting failed for every resample.")

    arr = np.array(bootstrap_paths)
    alpha = (1.0 - interval_level) / 2.0
    lower = np.quantile(arr, alpha, axis=0)
    upper = np.quantile(arr, 1.0 - alpha, axis=0)
    mean = np.mean(arr, axis=0)

    historical = pd.DataFrame(
        {
            "period": history["period"].to_numpy(),
            "actual": history_values,
            "prediction": np.nan,
            "lower": np.nan,
            "upper": np.nan,
            "is_forecast": False,
        }
    )
    future = pd.DataFrame(
        {
            "period": future_periods,
            "actual": np.nan,
            "prediction": mean,
            "lower": lower,
            "upper": upper,
            "is_forecast": True,
        }
    )
    predictions = pd.concat([historical, future], ignore_index=True)[PREDICTION_COLUMNS]

    meta = {
        "strategy": "autoregressive",
        "target_id": target_id,
        "model": "ridge",
        "lags": clean_lags,
        "n_bootstrap": len(bootstrap_paths),
        "interval_level": interval_level,
    }
    return predictions, meta


def forecast_target(
    series: pd.DataFrame,
    target_id: str,
    horizon_periods: int,
    config: PanelBuildConfig = PanelBuildConfig(),
    predictor_id: str | None = None,
    *,
    interval_level: float = 0.80,
    n_bootstrap: int = 25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Forecast a target past its last observation, picking the best available strategy.

    Uses the leading-indicator strategy when ``predictor_id`` is given and has
    rows dated after the target's last observation; falls back to an
    autoregressive projection of the target's own history otherwise.
    """
    if predictor_id:
        try:
            return forecast_with_leading_indicator(
                series,
                predictor_id,
                target_id,
                horizon_periods,
                freq=config.freq,
                lags=config.lags,
                aggregation=config.aggregation,
                interval_level=interval_level,
            )
        except ValueError as exc:
            predictions, meta = forecast_autoregressive(
                series,
                target_id,
                horizon_periods,
                config=config,
                interval_level=interval_level,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
            meta["requested_predictor_id"] = predictor_id
            meta["fallback_reason"] = str(exc)
            return predictions, meta

    return forecast_autoregressive(
        series,
        target_id,
        horizon_periods,
        config=config,
        interval_level=interval_level,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

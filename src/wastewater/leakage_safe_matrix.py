"""Leakage-safe pairwise respiratory admission forecasting.

This module is designed for notebook 05. It evaluates every predictive /
predicted pair using expanding-window, out-of-sample predictions only. Each
prediction at period t is fitted using rows strictly before t, and predictors are
restricted to positive lags so same-period or future values cannot leak into the
forecast.
"""
from __future__ import annotations

from math import erfc, sqrt
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def period_column(date: pd.Series, freq: str) -> pd.Series:
    """Convert dates to weekly or monthly period starts."""
    if freq.upper().startswith("M"):
        return pd.to_datetime(date).dt.to_period("M").dt.to_timestamp()
    return pd.to_datetime(date).dt.to_period("W").dt.start_time


def aggregate_one_series(df: pd.DataFrame, freq: str = "W", aggregation: str = "mean") -> pd.DataFrame:
    """Aggregate one canonical series to a modelling frequency."""
    out = df.copy()
    out["period"] = period_column(out["date"], freq)
    if aggregation == "sum":
        grouped = out.groupby("period", dropna=False)["value"].sum(min_count=1)
    else:
        grouped = out.groupby("period", dropna=False)["value"].mean()
    return grouped.reset_index(name="value").dropna(subset=["period", "value"]).sort_values("period")


def make_pair_frame(
    predictor: pd.DataFrame,
    target: pd.DataFrame,
    *,
    freq: str = "W",
    lags: Iterable[int] = (1, 2, 3, 4),
    aggregation: str = "mean",
) -> tuple[pd.DataFrame, list[str]]:
    """Create a pairwise supervised frame using positive predictor lags only."""
    clean_lags = sorted({int(lag) for lag in lags})
    if not clean_lags or any(lag <= 0 for lag in clean_lags):
        raise ValueError("Leakage-safe evaluation requires strictly positive lags, e.g. 1,2,3,4")

    x = aggregate_one_series(predictor, freq=freq, aggregation=aggregation).rename(columns={"value": "x"})
    y = aggregate_one_series(target, freq=freq, aggregation=aggregation).rename(columns={"value": "y"})
    frame = pd.merge(x, y, on="period", how="inner").sort_values("period").reset_index(drop=True)
    for lag in clean_lags:
        frame[f"x_lag{lag}"] = frame["x"].shift(lag)
    predictor_cols = [f"x_lag{lag}" for lag in clean_lags]
    return frame[["period", "y", *predictor_cols]].dropna().reset_index(drop=True), predictor_cols


def normalise_using_train(train_values: pd.DataFrame | pd.Series, values: pd.DataFrame | pd.Series):
    """Z-score using training-window mean and standard deviation only."""
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0).replace(0, np.nan) if isinstance(train_values, pd.DataFrame) else train_values.std()
    if isinstance(std, pd.Series):
        std = std.fillna(1.0)
    elif not np.isfinite(std) or std == 0:
        std = 1.0
    return (values - mean) / std, mean, std


def _normal_survival(x: float) -> float:
    """Survival function for standard normal without requiring scipy."""
    if not np.isfinite(x):
        return float("nan")
    return float(0.5 * erfc(x / sqrt(2.0)))


def _metrics(y_true: pd.Series, y_pred: pd.Series, baseline_pred: pd.Series) -> dict[str, float]:
    y_true = pd.Series(y_true, dtype="float64")
    y_pred = pd.Series(y_pred, dtype="float64")
    baseline_pred = pd.Series(baseline_pred, dtype="float64")
    err = y_true - y_pred
    baseline_err = y_true - baseline_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    baseline_rmse = float(np.sqrt(np.mean(baseline_err**2)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": rmse,
        "max_abs_error": float(np.max(np.abs(err))),
        "mean_abs_baseline_error": float(np.mean(np.abs(baseline_err))),
        "baseline_rmse": baseline_rmse,
        "mse_skill_vs_rolling_train_mean": float(1 - (rmse**2 / baseline_rmse**2)) if baseline_rmse > 0 else float("nan"),
        "r2": float(1 - np.sum(err**2) / denom) if denom > 0 else float("nan"),
        "correlation": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan"),
    }


def rolling_origin_pair_forecast(
    predictor: pd.DataFrame,
    target: pd.DataFrame,
    *,
    predictor_id: str,
    target_id: str,
    freq: str = "W",
    lags: Iterable[int] = (1, 2, 3, 4),
    aggregation: str = "mean",
    initial_train_fraction: float = 0.6,
    min_train_size: int = 12,
    ridge_alpha: float = 1.0,
    spike_quantile: float = 0.9,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Forecast one pair using expanding-window, leakage-safe predictions.

    For every predicted period t, the model is fitted only on rows with period < t.
    Features are positive-lag predictor values, standardised using the current
    training window only. No same-period or future predictor/target values are
    used to fit the prediction for t.
    """
    frame, predictor_cols = make_pair_frame(predictor, target, freq=freq, lags=lags, aggregation=aggregation)
    n = len(frame)
    if n < max(3, min_train_size + 1):
        raise ValueError(f"Not enough complete lagged rows for leakage-safe evaluation: {n}")

    start_idx = max(min_train_size, int(np.floor(n * initial_train_fraction)))
    start_idx = min(start_idx, n - 1)
    rows: list[dict[str, object]] = []

    for test_pos in range(start_idx, n):
        train = frame.iloc[:test_pos].copy()
        test = frame.iloc[[test_pos]].copy()
        train_end = train["period"].max()
        test_period = test["period"].iloc[0]
        if not train_end < test_period:
            raise AssertionError("Leakage guard failed: training period is not strictly before prediction period")

        X_train_raw = train[predictor_cols].astype(float)
        X_test_raw = test[predictor_cols].astype(float)
        y_train_raw = train["y"].astype(float)
        y_test = float(test["y"].iloc[0])

        X_train_z, x_mean, x_std = normalise_using_train(X_train_raw, X_train_raw)
        X_test_z = (X_test_raw - x_mean) / x_std
        y_train_z, y_mean, y_std = normalise_using_train(y_train_raw, y_train_raw)

        model = Ridge(alpha=ridge_alpha, fit_intercept=True)
        model.fit(X_train_z, y_train_z)
        pred_z = float(model.predict(X_test_z)[0])
        prediction = float(pred_z * y_std + y_mean)

        train_pred_z = model.predict(X_train_z)
        train_pred = train_pred_z * y_std + y_mean
        residual_sigma = float(np.std(y_train_raw.to_numpy() - train_pred, ddof=1)) if len(train) > 2 else float(y_train_raw.std())
        if not np.isfinite(residual_sigma) or residual_sigma <= 0:
            residual_sigma = float(y_train_raw.std()) if y_train_raw.std() > 0 else 1.0

        spike_threshold = float(y_train_raw.quantile(spike_quantile))
        spike_z = (spike_threshold - prediction) / residual_sigma
        spike_probability = _normal_survival(spike_z)

        rows.append(
            {
                "predictor_id": predictor_id,
                "target_id": target_id,
                "period": test_period,
                "train_start": train["period"].min(),
                "train_end": train_end,
                "n_train": len(train),
                "actual": y_test,
                "prediction": prediction,
                "baseline_prediction": float(y_train_raw.mean()),
                "last_observed_target": float(y_train_raw.iloc[-1]),
                "residual": y_test - prediction,
                "abs_error": abs(y_test - prediction),
                "spike_threshold": spike_threshold,
                "spike_probability": spike_probability,
                "spike_score": 100.0 * spike_probability,
                "actual_spike": bool(y_test >= spike_threshold),
                "lags": ",".join(str(int(lag)) for lag in sorted({int(lag) for lag in lags})),
                "model": "ridge_expanding_origin",
                "ridge_alpha": ridge_alpha,
            }
        )

    predictions = pd.DataFrame(rows)
    metrics = _metrics(predictions["actual"], predictions["prediction"], predictions["baseline_prediction"])
    latest = predictions.sort_values("period").iloc[-1]
    result = {
        "predictor_id": predictor_id,
        "target_id": target_id,
        "freq": freq,
        "lags": ",".join(str(int(lag)) for lag in sorted({int(lag) for lag in lags})),
        "model": "ridge_expanding_origin",
        "ridge_alpha": ridge_alpha,
        "n_complete": int(n),
        "n_predictions": int(len(predictions)),
        "min_train_size": int(min_train_size),
        "initial_train_fraction": float(initial_train_fraction),
        "first_prediction_period": predictions["period"].min(),
        "last_prediction_period": predictions["period"].max(),
        "latest_prediction": float(latest["prediction"]),
        "latest_actual": float(latest["actual"]),
        "latest_spike_threshold": float(latest["spike_threshold"]),
        "latest_spike_score": float(latest["spike_score"]),
        "latest_actual_spike": bool(latest["actual_spike"]),
        **metrics,
    }
    return result, predictions


def run_leakage_safe_pairwise_matrix(
    series: pd.DataFrame,
    *,
    freq: str = "W",
    lags: Iterable[int] = (1, 2, 3, 4),
    aggregation: str = "mean",
    initial_train_fraction: float = 0.6,
    min_train_size: int = 12,
    ridge_alpha: float = 1.0,
    spike_quantile: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run leakage-safe rolling-origin forecasts for every predictive/predicted pair."""
    if series.empty:
        return pd.DataFrame(), pd.DataFrame()

    predictive_ids = sorted(series.loc[series["role"] == "predictive", "series_id"].dropna().unique())
    predicted_ids = sorted(series.loc[series["role"] == "predicted", "series_id"].dropna().unique())
    metadata = (
        series[["series_id", "role", "dataset_family", "series_name", "source_file"]]
        .drop_duplicates("series_id")
        .set_index("series_id")
    )

    results: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for predictor_id in predictive_ids:
        predictor = series[series["series_id"] == predictor_id]
        for target_id in predicted_ids:
            target = series[series["series_id"] == target_id]
            try:
                result, pred = rolling_origin_pair_forecast(
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
                    spike_quantile=spike_quantile,
                )
                result.update(
                    {
                        "predictor_family": metadata.loc[predictor_id, "dataset_family"],
                        "predictor_name": metadata.loc[predictor_id, "series_name"],
                        "predictor_source_file": metadata.loc[predictor_id, "source_file"],
                        "target_family": metadata.loc[target_id, "dataset_family"],
                        "target_name": metadata.loc[target_id, "series_name"],
                        "target_source_file": metadata.loc[target_id, "source_file"],
                        "status": "ok",
                        "error": "",
                    }
                )
                results.append(result)
                predictions.append(pred)
            except Exception as exc:
                results.append(
                    {
                        "predictor_id": predictor_id,
                        "target_id": target_id,
                        "predictor_family": metadata.loc[predictor_id, "dataset_family"],
                        "predictor_name": metadata.loc[predictor_id, "series_name"],
                        "predictor_source_file": metadata.loc[predictor_id, "source_file"],
                        "target_family": metadata.loc[target_id, "dataset_family"],
                        "target_name": metadata.loc[target_id, "series_name"],
                        "target_source_file": metadata.loc[target_id, "source_file"],
                        "status": "error",
                        "error": repr(exc),
                    }
                )

    results_df = pd.DataFrame(results)
    if not results_df.empty and "rmse" in results_df.columns:
        results_df = results_df.sort_values("rmse", ascending=True, na_position="last")
    predictions_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return results_df, predictions_df


def best_worst_summary(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return best/worst pair summaries by several held-out criteria."""
    ok = results[results["status"] == "ok"].copy() if not results.empty and "status" in results.columns else pd.DataFrame()
    if ok.empty:
        return {"best_rmse": ok, "worst_rmse": ok, "best_skill": ok, "worst_abs_error": ok, "highest_spike_score": ok}
    return {
        "best_rmse": ok.sort_values("rmse", ascending=True).head(10),
        "worst_rmse": ok.sort_values("rmse", ascending=False).head(10),
        "best_skill": ok.sort_values("mse_skill_vs_rolling_train_mean", ascending=False).head(10),
        "worst_abs_error": ok.sort_values("max_abs_error", ascending=False).head(10),
        "highest_spike_score": ok.sort_values("latest_spike_score", ascending=False).head(10),
    }


__all__ = [
    "period_column",
    "aggregate_one_series",
    "make_pair_frame",
    "normalise_using_train",
    "rolling_origin_pair_forecast",
    "run_leakage_safe_pairwise_matrix",
    "best_worst_summary",
]

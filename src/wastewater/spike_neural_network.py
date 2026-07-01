"""Neural-network spike early-warning models for respiratory-virus series.

This module turns the repository's canonical long-format time series into a
leakage-aware supervised learning problem:

    given information available at week t, predict whether a respiratory-virus
target series will enter a spike state over the next h weeks, and estimate
the severity of that spike with a bootstrap uncertainty interval.

The implementation deliberately uses scikit-learn's MLP models rather than
PyTorch/TensorFlow so that the repository's current dependency set remains
sufficient. It is intended as a strong first neural baseline; a later TCN,
LSTM, or graph neural network can reuse the same spike-label construction and
evaluation tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .ml_panel import PanelBuildConfig, build_wide_panel, chronological_split
from .regression_matrix import build_available_series


@dataclass(frozen=True)
class SpikeNNConfig:
    """Configuration for spike-label construction and neural-network fitting.

    Parameters
    ----------
    freq:
        Aggregation frequency used before modelling. Weekly data are the
        default because respiratory surveillance sources are usually weekly.
    lags:
        Positive lags used as predictors. Lag 1 means the most recent completed
        period before the prediction origin. Do not include negative lags.
    horizons:
        Forecast horizons, in periods, over which the future maximum is tested
        for a spike. Horizon 4 asks whether a spike occurs in the next 4 weeks.
    smoothing_window:
        Rolling mean window applied to the target before thresholds and labels
        are constructed.
    threshold_quantile:
        Historical quantile, estimated on the training period only, used as the
        spike threshold. 0.85 means "top 15% of historically observed activity."
    severity_band_cutpoints:
        Cut points, in training-scale units above the spike threshold, used to
        turn a continuous severity score into ordered bands.
    interval_level:
        Central bootstrap interval level for the predicted severity score.
    decision_threshold:
        Probability threshold used when reporting binary spike-classification
        metrics.
    n_bootstrap_models:
        Number of bootstrap MLP regressors used for severity intervals. Larger
        values give smoother intervals at higher computational cost.
    """

    freq: str = "W"
    lags: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12)
    horizons: tuple[int, ...] = (1, 2, 3, 4)
    aggregation: str = "mean"
    min_non_missing_fraction: float = 0.2
    smoothing_window: int = 3
    threshold_quantile: float = 0.85
    severity_band_cutpoints: tuple[float, float, float] = (0.25, 1.0, 2.0)
    interval_level: float = 0.80
    decision_threshold: float = 0.50
    hidden_layer_sizes: tuple[int, ...] = (64, 32)
    alpha: float = 1.0e-3
    max_iter: int = 1500
    n_bootstrap_models: int = 25
    min_train_rows: int = 24
    min_test_rows: int = 4


def _safe_series_id(series_id: str) -> str:
    """Return a feature-name-safe version of a canonical series id."""

    return str(series_id).replace("::", "__").replace("/", "_").replace(" ", "_")


def _future_window(values: pd.Series, horizon: int) -> pd.Series:
    """Return the future maximum over periods t+1, ..., t+horizon."""

    shifted = [values.shift(-step) for step in range(1, horizon + 1)]
    return pd.concat(shifted, axis=1).max(axis=1, skipna=True)


def _future_window_end(periods: pd.Series, horizon: int) -> pd.Series:
    """Return the final period included in each future prediction window."""

    return periods.shift(-horizon)


def _severity_band(score: pd.Series | np.ndarray, cutpoints: Sequence[float]) -> pd.Series:
    """Map a non-negative severity score to none/mild/moderate/severe bands."""

    s = pd.Series(score, dtype="float64")
    c1, c2, c3 = cutpoints
    band = pd.Series("none", index=s.index, dtype="object")
    band[(s > 0) & (s < c1)] = "borderline"
    band[(s >= c1) & (s < c2)] = "mild"
    band[(s >= c2) & (s < c3)] = "moderate"
    band[s >= c3] = "severe"
    return band


def _target_scale(train_values: pd.Series) -> float:
    """Robust scale used to express spike severity above the threshold."""

    train_values = pd.to_numeric(train_values, errors="coerce").dropna()
    if train_values.empty:
        return 1.0
    q75 = float(train_values.quantile(0.75))
    q25 = float(train_values.quantile(0.25))
    iqr = q75 - q25
    if np.isfinite(iqr) and iqr > 0:
        return iqr
    std = float(train_values.std())
    if np.isfinite(std) and std > 0:
        return std
    return 1.0


def make_spike_classifier(config: SpikeNNConfig, random_state: int = 42) -> Pipeline:
    """Create a scikit-learn MLP classifier for spike probability."""

    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=config.hidden_layer_sizes,
                    activation="relu",
                    alpha=config.alpha,
                    max_iter=config.max_iter,
                    early_stopping=True,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_severity_regressor(config: SpikeNNConfig, random_state: int = 42) -> Pipeline:
    """Create a scikit-learn MLP regressor for continuous spike severity."""

    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=config.hidden_layer_sizes,
                    activation="relu",
                    alpha=config.alpha,
                    max_iter=config.max_iter,
                    early_stopping=True,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_spike_feature_panel(
    series: pd.DataFrame,
    target_id: str,
    horizon: int,
    config: SpikeNNConfig = SpikeNNConfig(),
) -> tuple[pd.DataFrame, list[str]]:
    """Build a leakage-safe supervised panel for one target and horizon.

    The returned frame contains only features available at prediction origin
    ``period``: positive lags of all predictive series plus positive lags of the
    target itself. Labels are the future maximum target value over the next
    ``horizon`` periods; spike thresholds are intentionally *not* attached here
    because they must be estimated on the training period only.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if series.empty:
        return pd.DataFrame(), []

    target_ids = set(series.loc[series["role"] == "predicted", "series_id"].dropna().astype(str))
    if target_id not in target_ids:
        raise ValueError(f"Target is not a predicted series: {target_id}")

    panel_config = PanelBuildConfig(
        freq=config.freq,
        lags=config.lags,
        aggregation=config.aggregation,
        min_non_missing_fraction=config.min_non_missing_fraction,
    )
    wide = build_wide_panel(series, config=panel_config)
    if wide.empty or target_id not in wide.columns:
        return pd.DataFrame(), []

    wide = wide.sort_values("period").reset_index(drop=True)
    target_smoothed = (
        pd.to_numeric(wide[target_id], errors="coerce")
        .rolling(config.smoothing_window, min_periods=1)
        .mean()
    )

    panel = pd.DataFrame(
        {
            "period": wide["period"],
            "target_id": target_id,
            "horizon": horizon,
            "target_value": pd.to_numeric(wide[target_id], errors="coerce"),
            "target_smoothed": target_smoothed,
            "future_peak": _future_window(target_smoothed, horizon=horizon),
            "future_window_end": _future_window_end(wide["period"], horizon=horizon),
        }
    )

    feature_cols: list[str] = []
    predictive_ids = sorted(series.loc[series["role"] == "predictive", "series_id"].dropna().astype(str).unique())
    feature_series = [*predictive_ids, target_id]

    for sid in feature_series:
        if sid not in wide.columns:
            continue
        safe_sid = _safe_series_id(sid)
        source = pd.to_numeric(wide[sid], errors="coerce")
        for lag in config.lags:
            if lag <= 0:
                raise ValueError("Only positive lags are allowed for spike features")
            col = f"{safe_sid}__lag{lag}"
            panel[col] = source.shift(lag)
            feature_cols.append(col)

    panel = panel.dropna(subset=["period", "future_peak", "future_window_end"]).sort_values("period").reset_index(drop=True)
    if not feature_cols:
        return panel, []

    non_missing_fraction = panel[feature_cols].notna().mean(axis=0)
    kept = non_missing_fraction[non_missing_fraction >= config.min_non_missing_fraction].index.tolist()
    return panel[["period", "target_id", "horizon", "target_value", "target_smoothed", "future_peak", "future_window_end", *kept]], kept


def _add_train_based_labels(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    config: SpikeNNConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Attach spike and severity labels using train-period threshold only."""

    train_values = pd.to_numeric(train["target_smoothed"], errors="coerce").dropna()
    if train_values.empty:
        raise ValueError("Cannot build spike threshold: no training target values")

    threshold = float(train_values.quantile(config.threshold_quantile))
    scale = _target_scale(train_values)

    def add_labels(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["spike_threshold"] = threshold
        out["severity_scale"] = scale
        out["spike"] = (pd.to_numeric(out["future_peak"], errors="coerce") >= threshold).astype(int)
        out["severity_score"] = np.maximum(0.0, (pd.to_numeric(out["future_peak"], errors="coerce") - threshold) / scale)
        out["severity_band"] = _severity_band(out["severity_score"], config.severity_band_cutpoints).to_numpy()
        return out

    return add_labels(train), add_labels(test), {"spike_threshold": threshold, "severity_scale": scale}


def _split_train_test_without_label_leakage(
    panel: pd.DataFrame,
    *,
    train_fraction: float,
    min_test_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split and remove training rows whose labels cross split."""

    initial_train, test = chronological_split(panel, train_fraction=train_fraction, min_test_size=min_test_size)
    train_end = initial_train["period"].max()
    train = initial_train[initial_train["future_window_end"] <= train_end].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _classification_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute rare-event spike-classification metrics robustly."""

    prediction = (probability >= threshold).astype(int)
    out = {
        "brier": float(brier_score_loss(y_true, probability)) if len(y_true) else float("nan"),
        "precision": float(precision_score(y_true, prediction, zero_division=0)) if len(y_true) else float("nan"),
        "recall": float(recall_score(y_true, prediction, zero_division=0)) if len(y_true) else float("nan"),
        "f1": float(f1_score(y_true, prediction, zero_division=0)) if len(y_true) else float("nan"),
    }
    if len(np.unique(y_true)) > 1:
        out["average_precision"] = float(average_precision_score(y_true, probability))
        out["roc_auc"] = float(roc_auc_score(y_true, probability))
    else:
        out["average_precision"] = float("nan")
        out["roc_auc"] = float("nan")
    return out


def _fit_classifier_or_constant(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    *,
    config: SpikeNNConfig,
    random_state: int,
) -> tuple[np.ndarray, str]:
    """Fit MLP classifier, or return a constant probability when labels collapse."""

    if len(np.unique(y_train)) < 2:
        constant = float(np.mean(y_train)) if len(y_train) else 0.0
        return np.full(len(X_test), constant, dtype=float), "constant_single_class"

    classifier = make_spike_classifier(config, random_state=random_state)
    classifier.fit(X_train, y_train)
    probability = classifier.predict_proba(X_test)[:, 1]
    return np.clip(probability.astype(float), 0.0, 1.0), "mlp_classifier"


def _bootstrap_severity_interval(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    *,
    config: SpikeNNConfig,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Fit bootstrap MLP regressors and return mean/lower/upper predictions."""

    if len(X_train) < 2:
        constant = float(np.mean(y_train)) if len(y_train) else 0.0
        pred = np.full(len(X_test), constant, dtype=float)
        return pred, pred, pred, "constant_too_few_rows"

    rng = np.random.default_rng(random_state)
    predictions: list[np.ndarray] = []
    base = make_severity_regressor(config, random_state=random_state)

    for i in range(config.n_bootstrap_models):
        sample_index = rng.integers(0, len(X_train), size=len(X_train))
        model = clone(base)
        # Set the nested MLP random state differently for each bootstrap model.
        model.set_params(model__random_state=random_state + i + 1)
        try:
            model.fit(X_train.iloc[sample_index], y_train[sample_index])
            predictions.append(np.maximum(0.0, model.predict(X_test)))
        except Exception:
            continue

    if not predictions:
        constant = float(np.mean(y_train)) if len(y_train) else 0.0
        pred = np.full(len(X_test), constant, dtype=float)
        return pred, pred, pred, "constant_regressor_failed"

    arr = np.vstack(predictions)
    alpha = (1.0 - config.interval_level) / 2.0
    lower = np.quantile(arr, alpha, axis=0)
    upper = np.quantile(arr, 1.0 - alpha, axis=0)
    mean = np.mean(arr, axis=0)
    return mean, lower, upper, f"bootstrap_mlp_regressor_{len(predictions)}"


def fit_spike_nn_for_target(
    series: pd.DataFrame,
    target_id: str,
    config: SpikeNNConfig = SpikeNNConfig(),
    *,
    train_fraction: float = 0.8,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit spike-probability and severity-interval models for one target.

    Returns
    -------
    results:
        One row per forecast horizon with metrics and fit status.
    predictions:
        Out-of-sample predictions with actual spike/severity labels, predicted
        spike probabilities, and bootstrap severity intervals.
    """

    result_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for horizon in config.horizons:
        try:
            panel, feature_cols = build_spike_feature_panel(series, target_id, horizon, config=config)
            if not feature_cols or panel.empty:
                result_rows.append(
                    {
                        "target_id": target_id,
                        "horizon": horizon,
                        "status": "skipped",
                        "error": "No usable features or labels",
                    }
                )
                continue

            train, test = _split_train_test_without_label_leakage(
                panel,
                train_fraction=train_fraction,
                min_test_size=config.min_test_rows,
            )
            if len(train) < config.min_train_rows or len(test) < config.min_test_rows:
                result_rows.append(
                    {
                        "target_id": target_id,
                        "horizon": horizon,
                        "status": "skipped",
                        "error": f"Too few rows after split: train={len(train)}, test={len(test)}",
                        "n_features": len(feature_cols),
                    }
                )
                continue

            train, test, threshold_info = _add_train_based_labels(train, test, config=config)
            X_train = train[feature_cols]
            X_test = test[feature_cols]
            y_spike_train = train["spike"].astype(int).to_numpy()
            y_spike_test = test["spike"].astype(int).to_numpy()
            y_severity_train = train["severity_score"].astype(float).to_numpy()
            y_severity_test = test["severity_score"].astype(float).to_numpy()

            spike_probability, classifier_status = _fit_classifier_or_constant(
                X_train,
                y_spike_train,
                X_test,
                config=config,
                random_state=random_state + horizon * 100,
            )
            sev_mean, sev_lower, sev_upper, regressor_status = _bootstrap_severity_interval(
                X_train,
                y_severity_train,
                X_test,
                config=config,
                random_state=random_state + horizon * 1000,
            )

            class_metrics = _classification_metrics(
                y_spike_test,
                spike_probability,
                threshold=config.decision_threshold,
            )
            severity_mae = float(mean_absolute_error(y_severity_test, sev_mean))
            interval_coverage = float(np.mean((y_severity_test >= sev_lower) & (y_severity_test <= sev_upper)))
            interval_width = float(np.mean(sev_upper - sev_lower))

            result_rows.append(
                {
                    "target_id": target_id,
                    "horizon": horizon,
                    "status": "ok",
                    "classifier": classifier_status,
                    "severity_model": regressor_status,
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_features": len(feature_cols),
                    "train_start": train["period"].min(),
                    "train_end": train["period"].max(),
                    "test_start": test["period"].min(),
                    "test_end": test["period"].max(),
                    "train_spike_rate": float(np.mean(y_spike_train)),
                    "test_spike_rate": float(np.mean(y_spike_test)),
                    "severity_mae": severity_mae,
                    "severity_interval_coverage": interval_coverage,
                    "severity_interval_width": interval_width,
                    **threshold_info,
                    **class_metrics,
                }
            )

            pred = test[
                [
                    "period",
                    "target_id",
                    "horizon",
                    "target_value",
                    "target_smoothed",
                    "future_peak",
                    "future_window_end",
                    "spike_threshold",
                    "severity_scale",
                    "spike",
                    "severity_score",
                    "severity_band",
                ]
            ].copy()
            pred["spike_probability"] = spike_probability
            pred["predicted_spike"] = (pred["spike_probability"] >= config.decision_threshold).astype(int)
            pred["predicted_severity_score"] = sev_mean
            pred["severity_score_lower"] = sev_lower
            pred["severity_score_upper"] = sev_upper
            pred["predicted_severity_band"] = _severity_band(sev_mean, config.severity_band_cutpoints).to_numpy()
            pred["severity_band_lower"] = _severity_band(sev_lower, config.severity_band_cutpoints).to_numpy()
            pred["severity_band_upper"] = _severity_band(sev_upper, config.severity_band_cutpoints).to_numpy()
            pred["interval_level"] = config.interval_level
            prediction_frames.append(pred)

        except Exception as exc:
            result_rows.append(
                {
                    "target_id": target_id,
                    "horizon": horizon,
                    "status": "error",
                    "error": repr(exc),
                }
            )

    results = pd.DataFrame(result_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return results, predictions


def run_spike_neural_network_experiment(
    series: pd.DataFrame,
    target_ids: Sequence[str] | None = None,
    config: SpikeNNConfig = SpikeNNConfig(),
    *,
    train_fraction: float = 0.8,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit spike neural networks for all selected predicted target series."""

    if series.empty:
        return pd.DataFrame(), pd.DataFrame()

    if target_ids is None:
        target_ids = sorted(series.loc[series["role"] == "predicted", "series_id"].dropna().astype(str).unique())

    result_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    for target_id in target_ids:
        result, pred = fit_spike_nn_for_target(
            series,
            target_id=str(target_id),
            config=config,
            train_fraction=train_fraction,
            random_state=random_state,
        )
        if not result.empty:
            result_parts.append(result)
        if not pred.empty:
            prediction_parts.append(pred)

    results = pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame()
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    return results, predictions


def run_spike_neural_network_from_repo(
    root: str | Path,
    target_ids: Sequence[str] | None = None,
    config: SpikeNNConfig = SpikeNNConfig(),
    *,
    train_fraction: float = 0.8,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load available repository series and run the spike neural-network experiment."""

    series = build_available_series(Path(root))
    return run_spike_neural_network_experiment(
        series,
        target_ids=target_ids,
        config=config,
        train_fraction=train_fraction,
        random_state=random_state,
    )


def save_spike_outputs(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    root: str | Path,
    *,
    result_filename: str = "respiratory_spike_neural_network_results.csv",
    prediction_filename: str = "respiratory_spike_neural_network_predictions.csv",
) -> tuple[Path, Path]:
    """Write spike-model metrics and predictions under ``data/processed``."""

    processed = Path(root) / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    result_path = processed / result_filename
    prediction_path = processed / prediction_filename
    results.to_csv(result_path, index=False)
    predictions.to_csv(prediction_path, index=False)
    return result_path, prediction_path


__all__ = [
    "SpikeNNConfig",
    "build_spike_feature_panel",
    "fit_spike_nn_for_target",
    "run_spike_neural_network_experiment",
    "run_spike_neural_network_from_repo",
    "save_spike_outputs",
    "make_spike_classifier",
    "make_severity_regressor",
]

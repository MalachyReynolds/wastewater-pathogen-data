"""Panel construction and machine-learning train/test models for respiratory incidence.

This module uses the canonical series produced by ``wastewater.regression_matrix``
and builds a wide modelling panel with lagged predictive variables and one or
more predicted outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PanelBuildConfig:
    """Configuration for lagged panel construction."""

    freq: str = "W"
    lags: tuple[int, ...] = (1, 2, 3, 4)
    aggregation: str = "mean"
    min_non_missing_fraction: float = 0.2


def _period_column(date: pd.Series, freq: str) -> pd.Series:
    if freq.upper().startswith("M"):
        return pd.to_datetime(date).dt.to_period("M").dt.to_timestamp()
    return pd.to_datetime(date).dt.to_period("W").dt.start_time


def _aggregate_series(series: pd.DataFrame, freq: str, aggregation: str) -> pd.DataFrame:
    df = series.copy()
    df["period"] = _period_column(df["date"], freq)
    if aggregation == "sum":
        grouped = df.groupby(["period", "series_id"], dropna=False)["value"].sum(min_count=1)
    else:
        grouped = df.groupby(["period", "series_id"], dropna=False)["value"].mean()
    return grouped.reset_index()


def build_wide_panel(series: pd.DataFrame, config: PanelBuildConfig = PanelBuildConfig()) -> pd.DataFrame:
    """Build a wide period-indexed panel of all predictive and predicted series."""
    if series.empty:
        return pd.DataFrame()

    aggregated = _aggregate_series(series, config.freq, config.aggregation)
    wide = aggregated.pivot(index="period", columns="series_id", values="value").sort_index()
    wide.columns = [str(c) for c in wide.columns]
    return wide.reset_index()


def build_lagged_feature_panel(
    series: pd.DataFrame,
    target_id: str,
    config: PanelBuildConfig = PanelBuildConfig(),
) -> tuple[pd.DataFrame, list[str]]:
    """Build a supervised panel for one target using lagged predictive variables."""
    if series.empty:
        return pd.DataFrame(), []

    predictive_ids = sorted(series.loc[series["role"] == "predictive", "series_id"].dropna().unique())
    if target_id not in set(series.loc[series["role"] == "predicted", "series_id"].dropna().unique()):
        raise ValueError(f"Target is not a predicted series: {target_id}")

    wide = build_wide_panel(series, config=config)
    if wide.empty or target_id not in wide.columns:
        return pd.DataFrame(), []

    panel = wide[["period", target_id]].copy().rename(columns={target_id: "target"})
    feature_cols: list[str] = []
    for sid in predictive_ids:
        if sid not in wide.columns:
            continue
        safe_sid = sid.replace("::", "__").replace("/", "_").replace(" ", "_")
        for lag in config.lags:
            col = f"{safe_sid}__lag{lag}"
            panel[col] = wide[sid].shift(lag)
            feature_cols.append(col)

    panel = panel.dropna(subset=["target"]).sort_values("period").reset_index(drop=True)
    if not feature_cols:
        return panel, []

    non_missing_fraction = panel[feature_cols].notna().mean(axis=0)
    kept = non_missing_fraction[non_missing_fraction >= config.min_non_missing_fraction].index.tolist()
    panel = panel[["period", "target", *kept]]
    return panel, kept


def chronological_split(
    panel: pd.DataFrame,
    train_fraction: float = 0.8,
    min_test_size: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a supervised panel into earlier train and later test rows."""
    panel = panel.sort_values("period").reset_index(drop=True)
    n = len(panel)
    if n < 3:
        raise ValueError(f"Need at least three target observations; found {n}")
    if n <= min_test_size:
        split_idx = max(1, n - 1)
    else:
        split_idx = int(np.floor(n * train_fraction))
        split_idx = min(split_idx, n - min_test_size)
        split_idx = max(1, split_idx)
    return panel.iloc[:split_idx].copy(), panel.iloc[split_idx:].copy()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, baseline: float) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_true, np.full_like(y_true, baseline, dtype=float))))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "correlation": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan"),
        "baseline_rmse": baseline_rmse,
        "mse_skill_vs_train_mean": float(1 - (rmse**2 / baseline_rmse**2)) if baseline_rmse > 0 else float("nan"),
    }


def model_specs(random_state: int = 42) -> dict[str, object]:
    """Return baseline machine-learning model specifications."""
    linear_pipeline = lambda model: Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
    tree_pipeline = lambda model: Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )
    return {
        "ols": linear_pipeline(LinearRegression()),
        "ridge": linear_pipeline(RidgeCV(alphas=np.logspace(-4, 4, 25))),
        "elastic_net": linear_pipeline(ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-4, 2, 20), max_iter=10000)),
        "random_forest": tree_pipeline(RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=random_state)),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=200, l2_regularization=1.0, random_state=random_state),
    }


def evaluate_models_for_target(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    target_id: str,
    train_fraction: float = 0.8,
    min_test_size: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate several models for one target under a chronological split."""
    if not feature_cols:
        return pd.DataFrame(), pd.DataFrame()

    complete_target = panel.dropna(subset=["target"]).copy()
    train, test = chronological_split(complete_target, train_fraction=train_fraction, min_test_size=min_test_size)
    if len(train) < 3 or len(test) < 1:
        raise ValueError("Train/test split is too small")

    X_train = train[list(feature_cols)]
    y_train = train["target"].astype(float).to_numpy()
    X_test = test[list(feature_cols)]
    y_test = test["target"].astype(float).to_numpy()
    baseline = float(np.mean(y_train))

    results: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for model_name, model in model_specs(random_state=random_state).items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metric = _metrics(y_test, y_pred, baseline=baseline)
            results.append(
                {
                    "target_id": target_id,
                    "model": model_name,
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_features": len(feature_cols),
                    "train_start": train["period"].min(),
                    "train_end": train["period"].max(),
                    "test_start": test["period"].min(),
                    "test_end": test["period"].max(),
                    "status": "ok",
                    "error": "",
                    **metric,
                }
            )
            pred_df = test[["period", "target"]].copy()
            pred_df["target_id"] = target_id
            pred_df["model"] = model_name
            pred_df["prediction"] = y_pred
            pred_df["baseline_prediction"] = baseline
            pred_df["residual"] = pred_df["target"] - pred_df["prediction"]
            predictions.append(pred_df)
        except Exception as exc:
            results.append(
                {
                    "target_id": target_id,
                    "model": model_name,
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_features": len(feature_cols),
                    "status": "error",
                    "error": repr(exc),
                }
            )

    return pd.DataFrame(results), pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def evaluate_all_targets(
    series: pd.DataFrame,
    config: PanelBuildConfig = PanelBuildConfig(),
    train_fraction: float = 0.8,
    min_test_size: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all predicted targets using all available predictive variables."""
    target_ids = sorted(series.loc[series["role"] == "predicted", "series_id"].dropna().unique())
    results: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    for target_id in target_ids:
        try:
            panel, feature_cols = build_lagged_feature_panel(series, target_id=target_id, config=config)
            result, pred = evaluate_models_for_target(
                panel,
                feature_cols,
                target_id=target_id,
                train_fraction=train_fraction,
                min_test_size=min_test_size,
                random_state=random_state,
            )
            results.append(result)
            if not pred.empty:
                predictions.append(pred)
        except Exception as exc:
            results.append(pd.DataFrame([{"target_id": target_id, "model": "all", "status": "error", "error": repr(exc)}]))
    return (
        pd.concat(results, ignore_index=True) if results else pd.DataFrame(),
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
    )

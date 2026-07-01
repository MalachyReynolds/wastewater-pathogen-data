from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


def discover_datasets(root: Path, limit: int = 200) -> List[Path]:
    """Discover likely tabular datasets inside the workspace."""
    allowed_suffixes = {".csv", ".parquet", ".xlsx", ".xls"}
    excluded_dirs = {".git", ".venv", "__pycache__", ".ipynb_checkpoints"}

    candidates: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in excluded_dirs for part in path.parts):
            continue
        if path.suffix.lower() in allowed_suffixes:
            candidates.append(path)

    candidates = sorted({path.resolve() for path in candidates})
    return candidates[:limit]


def load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def build_lag_features(frame: pd.DataFrame, columns: Iterable[str], lag_count: int) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns or not pd.api.types.is_numeric_dtype(out[column]):
            continue
        for lag in range(1, lag_count + 1):
            out[f"{column}_lag_{lag}"] = out[column].shift(lag)
    return out


def prepare_training_frame(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: Iterable[str],
    date_column: str | None = None,
    use_lags: bool = False,
    lag_count: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    working = frame.copy()

    if date_column and date_column in working.columns:
        working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
        working = working.sort_values(date_column).reset_index(drop=True)

    selected_feature_columns = [col for col in feature_columns if col in working.columns and col != target_column]
    if use_lags:
        working = build_lag_features(working, selected_feature_columns, lag_count)
        selected_feature_columns = [col for col in working.columns if col in selected_feature_columns or col.startswith(tuple(f"{c}_lag_" for c in selected_feature_columns))]

    numeric_features = [
        column
        for column in selected_feature_columns
        if pd.api.types.is_numeric_dtype(working[column]) and column != target_column
    ]

    if not numeric_features:
        raise ValueError("No numeric feature columns were available for modelling.")

    modeling_frame = working[numeric_features + [target_column]].copy()
    if date_column and date_column in modeling_frame.columns:
        modeling_frame = modeling_frame.drop(columns=[date_column])

    modeling_frame = modeling_frame.apply(pd.to_numeric, errors="coerce")
    modeling_frame = modeling_frame.dropna(subset=[target_column])

    feature_frame = modeling_frame[numeric_features]
    target_series = modeling_frame[target_column]
    mask = feature_frame.notna().all(axis=1) & target_series.notna()

    return modeling_frame.loc[mask].reset_index(drop=True), feature_frame.loc[mask].reset_index(drop=True), target_series.loc[mask].reset_index(drop=True)


def fit_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    test_size: float = 0.25,
) -> dict:
    if len(X) < 10:
        raise ValueError("At least 10 rows are required to train a model.")

    split_index = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if len(X_train) < 5 or len(X_test) < 3:
        raise ValueError("The selected dataset is too small for a meaningful train/test split.")

    if model_name == "linear":
        estimator = LinearRegression()
    elif model_name == "ridge":
        estimator = Ridge(alpha=1.0)
    else:
        estimator = RandomForestRegressor(n_estimators=200, random_state=42)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": r2_score(y_test, predictions),
    }

    if hasattr(pipeline.named_steps["model"], "coef_"):
        importances = pd.Series(pipeline.named_steps["model"].coef_, index=X.columns)
    elif hasattr(pipeline.named_steps["model"], "feature_importances_"):
        importances = pd.Series(pipeline.named_steps["model"].feature_importances_, index=X.columns)
    else:
        importances = pd.Series(np.zeros(len(X.columns)), index=X.columns)

    return {
        "pipeline": pipeline,
        "predictions": predictions,
        "y_test": y_test,
        "metrics": metrics,
        "importances": importances.sort_values(ascending=False),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def summarise_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.describe(include="all").transpose()
    summary["missing"] = frame.isna().sum()
    summary["missing_pct"] = (summary["missing"] / len(frame) * 100).round(2)
    return summary


def plot_correlation(frame: pd.DataFrame, numeric_columns: List[str]) -> plt.Figure:
    correlation = frame[numeric_columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(correlation, cmap="viridis")
    ax.set_xticks(range(len(numeric_columns)))
    ax.set_xticklabels(numeric_columns, rotation=45, ha="right")
    ax.set_yticks(range(len(numeric_columns)))
    ax.set_yticklabels(numeric_columns)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    return fig

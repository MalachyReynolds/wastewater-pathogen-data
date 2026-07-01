"""Regression helpers for Google Trends search-term time-series files.

The one-year search trend files used for the main regression live in
``Google_trends_v2/1y_data`` and are expected to have filenames beginning with
``time_series_GB``. The search-volume predictor is always the second source
column by position, regardless of the column name.

These helpers scan a local checkout, load the files, coerce that second column
to a canonical ``count`` field, and build regression frames against UKHSA
GP/admission chart series.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .ukhsa import build_ukhsa_series_catalogue, chart_to_series

DEFAULT_SEARCH_TERMS_DIR = Path("Google_trends_v2") / "1y_data"
DEFAULT_SEARCH_TERMS_PATTERN = "time_series_GB*"


def normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalised lowercase column names."""
    out = df.copy()
    out.columns = (
        pd.Index(out.columns)
        .astype(str)
        .str.strip()
        .str.replace(r"[^0-9A-Za-z]+", "_", regex=True)
        .str.strip("_")
        .str.lower()
    )
    return out


def find_search_term_files(
    root: Path,
    search_dir: str | Path | None = DEFAULT_SEARCH_TERMS_DIR,
    pattern: str = DEFAULT_SEARCH_TERMS_PATTERN,
) -> pd.DataFrame:
    """Find one-year Google Trends search-term files.

    By default this searches only ``Google_trends_v2/1y_data``. Pass
    ``search_dir=None`` to scan the whole repository checkout.
    """
    root = Path(root).resolve()
    base = root if search_dir is None else root / Path(search_dir)
    files = sorted(
        p for p in base.rglob(pattern)
        if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".txt", ".json"}
    ) if base.exists() else []
    return pd.DataFrame(
        {
            "path": [p.relative_to(root).as_posix() for p in files],
            "filename": [p.name for p in files],
            "search_dir": [Path(search_dir).as_posix() if search_dir is not None else "." for _ in files],
            "suffix": [p.suffix.lower() for p in files],
            "size_kb": [p.stat().st_size / 1024 for p in files],
        }
    )


def read_search_term_file(path: Path) -> pd.DataFrame:
    """Read a search-term time-series file with tolerant delimiter handling."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        return normalise_column_names(pd.read_json(path))
    if path.suffix.lower() == ".tsv":
        return normalise_column_names(pd.read_csv(path, sep="\t"))
    try:
        return normalise_column_names(pd.read_csv(path, encoding="utf-8-sig"))
    except UnicodeDecodeError:
        return normalise_column_names(pd.read_csv(path, encoding="latin1"))
    except Exception:
        return normalise_column_names(pd.read_csv(path, sep=";"))


def infer_date_column(df: pd.DataFrame) -> str:
    """Infer the date column in a search-term file."""
    candidates = ["date", "week", "month", "period", "time", "timestamp", "x"]
    for col in candidates:
        if col in df.columns and pd.to_datetime(df[col], errors="coerce").notna().any():
            return col
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().sum() >= max(3, len(df) // 3):
            return col
    raise ValueError(f"Could not infer date column from columns={list(df.columns)}")


def second_source_column(df: pd.DataFrame, rel_path: str, value_column_index: int = 1) -> str:
    """Return the predictor column selected by source-column position.

    By default, ``value_column_index=1`` selects the second source column after
    reading and column normalisation. This matches the search-trends exports in
    this repository, where the predictive count field is always the second
    column but may have different names across files.
    """
    if len(df.columns) <= value_column_index:
        raise ValueError(
            f"Expected at least {value_column_index + 1} columns in {rel_path}; "
            f"available columns={list(df.columns)}"
        )
    return str(df.columns[value_column_index])


def infer_search_term_from_filename(path: Path) -> str:
    """Infer a readable search-term label from a time_series_GB filename."""
    stem = Path(path).stem
    stem = re.sub(r"^time[_-]series[_-]GB[_-]?", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[_-]+", " ", stem).strip()
    return stem or Path(path).stem


def search_file_to_long(root: Path, rel_path: str, value_column_index: int = 1) -> pd.DataFrame:
    """Convert one search-term file to long format using its second column.

    The selected source column is stored in ``value_column`` for traceability,
    but the canonical predictor column is always named ``count`` downstream.
    """
    path = Path(root) / rel_path
    df = read_search_term_file(path)
    date_col = infer_date_column(df)
    value_col = second_source_column(df, rel_path, value_column_index=value_column_index)

    values = pd.to_numeric(df[value_col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "search_term": infer_search_term_from_filename(path),
            "count": values,
            "source_file": rel_path,
            "date_column": date_col,
            "value_column": value_col,
            "value_column_index": value_column_index,
        }
    ).dropna(subset=["date", "count"])

    out["week"] = out["date"].dt.to_period("W").dt.start_time
    out["month"] = out["date"].dt.to_period("M").dt.to_timestamp()
    return out


def build_search_term_catalogue(
    root: Path,
    search_dir: str | Path | None = DEFAULT_SEARCH_TERMS_DIR,
    value_column_index: int = 1,
) -> pd.DataFrame:
    """Return file metadata plus inferred schema for every selected search-term file."""
    files = find_search_term_files(root, search_dir=search_dir)
    rows: list[dict] = []
    for row in files.to_dict(orient="records"):
        try:
            df = read_search_term_file(Path(root) / row["path"])
            date_col = infer_date_column(df)
            value_col = second_source_column(df, row["path"], value_column_index=value_column_index)
            values = pd.to_numeric(df[value_col].astype(str).str.replace(",", "", regex=False), errors="coerce")
            usable_values = int(values.notna().sum())
            rows.append(
                {
                    **row,
                    "date_column": date_col,
                    "predictor_column_index": value_column_index,
                    "predictor_column": value_col,
                    "usable_predictor_values": usable_values,
                    "columns": list(df.columns),
                    "status": "ok" if usable_values > 0 else "error",
                    "error": "" if usable_values > 0 else f"Second column '{value_col}' has no numeric values",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **row,
                    "date_column": "",
                    "predictor_column_index": value_column_index,
                    "predictor_column": "",
                    "usable_predictor_values": 0,
                    "columns": [],
                    "status": "error",
                    "error": repr(exc),
                }
            )
    return pd.DataFrame(rows)


def standardise(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    std = values.std(skipna=True)
    if pd.isna(std) or std == 0:
        return values * np.nan
    return (values - values.mean(skipna=True)) / std


def build_gp_admissions_series_from_ukhsa(
    root: Path,
    outcome_files: Sequence[str] | None = None,
    freq: str = "W",
) -> pd.DataFrame:
    """Build a GP/admissions outcome series from UKHSA chart files.

    If outcome_files is omitted, files inferred as ``gp_admissions`` by
    ``wastewater.ukhsa`` are used. Pass outcome_files explicitly when automatic
    inference is wrong.
    """
    root = Path(root)
    catalogue = build_ukhsa_series_catalogue(root)
    if outcome_files is None:
        outcome_files = catalogue.loc[catalogue["series_type"] == "gp_admissions", "path"].tolist()
    if not outcome_files:
        raise ValueError("No UKHSA GP/admission outcome files found. Pass outcome_files explicitly.")

    outcome = pd.concat([chart_to_series(root, p, series_type="gp_admissions") for p in outcome_files], ignore_index=True)
    period_col = "week" if freq.upper().startswith("W") else "month"
    return outcome.groupby(period_col, dropna=False)["value"].sum(min_count=1).reset_index(name="gp_admissions")


def _safe_predictor_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_").lower()


def build_search_term_regression_frame(
    root: Path,
    search_files: Sequence[str] | None = None,
    outcome_files: Sequence[str] | None = None,
    search_dir: str | Path | None = DEFAULT_SEARCH_TERMS_DIR,
    freq: str = "W",
    lags: Iterable[int] = (0, 1, 2, 3, 4),
    aggregate_terms: bool = False,
    value_column_index: int = 1,
) -> pd.DataFrame:
    """Build a lagged regression frame for search-term counts vs GP admissions.

    Parameters
    ----------
    search_dir:
        Directory to scan for search term files when ``search_files`` is not
        supplied. Defaults to ``Google_trends_v2/1y_data``.
    aggregate_terms:
        If True, all second-column values are summed into ``search_count`` before
        lagging. If False, each one-year Google Trends file is pivoted into a
        separate predictor, still using the second source column from each file.
    value_column_index:
        Zero-based source-column index to use as the search-volume predictor.
        The default, 1, means the second column.
    """
    root = Path(root)
    if search_files is None:
        search_files = find_search_term_files(root, search_dir=search_dir)["path"].tolist()
    if not search_files:
        raise ValueError("No search-term files matching time_series_GB* were found in the selected directory.")

    terms = pd.concat([search_file_to_long(root, p, value_column_index=value_column_index) for p in search_files], ignore_index=True)
    period_col = "week" if freq.upper().startswith("W") else "month"
    outcome = build_gp_admissions_series_from_ukhsa(root, outcome_files=outcome_files, freq=freq)

    if aggregate_terms:
        x = terms.groupby(period_col, dropna=False)["count"].sum(min_count=1).reset_index(name="search_count")
        frame = pd.merge(x, outcome, on=period_col, how="inner").rename(columns={period_col: "period"}).sort_values("period")
        frame["z_search_count"] = standardise(frame["search_count"])
        frame["z_gp_admissions"] = standardise(frame["gp_admissions"])
        for lag in lags:
            frame[f"z_search_count_lag{lag}"] = frame["z_search_count"].shift(lag)
        return frame

    pivot = (
        terms.groupby([period_col, "search_term"], dropna=False)["count"]
        .sum(min_count=1)
        .reset_index()
        .pivot(index=period_col, columns="search_term", values="count")
        .reset_index()
    )
    frame = pd.merge(pivot, outcome, on=period_col, how="inner").rename(columns={period_col: "period"}).sort_values("period")
    frame["z_gp_admissions"] = standardise(frame["gp_admissions"])
    raw_predictor_cols = [c for c in frame.columns if c not in {"period", "gp_admissions", "z_gp_admissions"}]
    for col in raw_predictor_cols:
        zcol = "z_" + _safe_predictor_name(col)
        frame[zcol] = standardise(frame[col])
        for lag in lags:
            frame[f"{zcol}_lag{lag}"] = frame[zcol].shift(lag)
    return frame


def search_term_lagged_predictor_columns(
    frame: pd.DataFrame,
    lags: Iterable[int] = (0, 1, 2, 3, 4),
) -> list[str]:
    """Return lagged Google Trends predictor columns from a regression frame."""
    lag_suffixes = tuple(f"_lag{lag}" for lag in lags)
    excluded_prefixes = ("z_gp_admissions",)
    return sorted(
        col for col in frame.columns
        if col.startswith("z_") and col.endswith(lag_suffixes) and not col.startswith(excluded_prefixes)
    )


def _design_matrix(
    df: pd.DataFrame,
    predictor_columns: Sequence[str],
    seasonal_controls: bool = True,
    time_col: str = "period",
) -> pd.DataFrame:
    """Construct a regression design matrix with optional month dummies."""
    X = df[list(predictor_columns)].copy()
    if seasonal_controls:
        month = pd.to_datetime(df[time_col]).dt.month.astype("category")
        X = pd.concat([X, pd.get_dummies(month, prefix="month", drop_first=True, dtype=float)], axis=1)
    return sm.add_constant(X, has_constant="add")


def fit_search_term_ols(
    frame: pd.DataFrame,
    lags: Iterable[int] = (0, 1, 2, 3, 4),
    predictor_prefix: str = "z_search_count_lag",
    predictor_columns: Sequence[str] | None = None,
    seasonal_controls: bool = True,
):
    """Fit OLS for z-scored GP admissions on search-term predictors.

    If ``predictor_columns`` is supplied, those columns are used directly. This
    is the preferred mode for the one-year Google Trends files, where each file
    is a separate predictive variable. Otherwise, the legacy aggregate predictor
    prefix is used.
    """
    predictors = list(predictor_columns) if predictor_columns is not None else [f"{predictor_prefix}{lag}" for lag in lags]
    model_df = frame[["period", "z_gp_admissions", *predictors]].dropna().copy()
    y = model_df["z_gp_admissions"]
    X = _design_matrix(model_df, predictors, seasonal_controls=seasonal_controls)
    maxlags = max(lags) if lags else 1
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


def chronological_train_test_split(
    frame: pd.DataFrame,
    predictor_columns: Sequence[str],
    outcome_col: str = "z_gp_admissions",
    time_col: str = "period",
    train_fraction: float = 0.8,
    min_test_size: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-series regression frame chronologically.

    Rows with missing outcome or predictors are dropped before splitting. The
    first block is the training set and the final block is the held-out test set.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie between 0 and 1")

    model_df = frame[[time_col, outcome_col, *predictor_columns]].dropna().sort_values(time_col).copy()
    n = len(model_df)
    if n < 3:
        raise ValueError(f"Need at least 3 complete observations for train/test split; found {n}")

    if n <= min_test_size:
        split_idx = max(1, n - 1)
    else:
        split_idx = int(np.floor(n * train_fraction))
        split_idx = min(split_idx, n - min_test_size)
        split_idx = max(1, split_idx)

    train = model_df.iloc[:split_idx].copy()
    test = model_df.iloc[split_idx:].copy()
    return train, test


def regression_metrics(y_true: pd.Series, y_pred: pd.Series, baseline_pred: float | None = None) -> dict[str, float]:
    """Compute predictive metrics for held-out regression predictions."""
    y_true = pd.Series(y_true, dtype="float64")
    y_pred = pd.Series(y_pred, dtype="float64")
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - np.sum(err**2) / denom) if denom > 0 else float("nan")
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")

    metrics: dict[str, float] = {
        "n_test": float(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "correlation": corr,
    }
    if baseline_pred is not None:
        baseline_err = y_true - baseline_pred
        baseline_rmse = float(np.sqrt(np.mean(baseline_err**2)))
        metrics["baseline_rmse"] = baseline_rmse
        metrics["mse_skill_vs_train_mean"] = float(1 - (rmse**2 / baseline_rmse**2)) if baseline_rmse > 0 else float("nan")
    return metrics


def fit_search_term_train_test(
    frame: pd.DataFrame,
    predictor_columns: Sequence[str],
    lags: Iterable[int] = (0, 1, 2, 3, 4),
    train_fraction: float = 0.8,
    min_test_size: int = 4,
    seasonal_controls: bool = False,
) -> dict[str, object]:
    """Fit on an earlier training window and evaluate on a later test window."""
    train, test = chronological_train_test_split(
        frame,
        predictor_columns=predictor_columns,
        train_fraction=train_fraction,
        min_test_size=min_test_size,
    )

    X_train = _design_matrix(train, predictor_columns, seasonal_controls=seasonal_controls)
    X_test = _design_matrix(test, predictor_columns, seasonal_controls=seasonal_controls)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    y_train = train["z_gp_admissions"]
    y_test = test["z_gp_admissions"]
    maxlags = max(lags) if lags else 1
    model = sm.OLS(y_train, X_train).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    test_pred = pd.Series(model.predict(X_test), index=test.index, name="prediction")
    train_mean = float(y_train.mean())
    metrics = regression_metrics(y_test, test_pred, baseline_pred=train_mean)

    train_out = train.copy()
    train_out["split"] = "train"
    train_out["prediction"] = pd.Series(model.predict(X_train), index=train.index)
    train_out["residual"] = train_out["z_gp_admissions"] - train_out["prediction"]

    test_out = test.copy()
    test_out["split"] = "test"
    test_out["prediction"] = test_pred
    test_out["baseline_prediction"] = train_mean
    test_out["residual"] = test_out["z_gp_admissions"] - test_out["prediction"]

    return {
        "model": model,
        "train": train_out,
        "test": test_out,
        "metrics": metrics,
        "predictor_columns": list(predictor_columns),
    }

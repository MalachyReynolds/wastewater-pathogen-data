"""Train-test regression matrix across predictive and predicted datasets.

This module builds a common catalogue of time series from the repository and
runs chronological train-test regressions for every compatible predictive /
predicted pair.

Current automatic sources
-------------------------
Predictive:
- Google Trends one-year files in ``Google_trends_v2/1y_data/time_series_GB*``
- UKHSA dashboard files classified as NHS-call series
- raw wastewater files listed in ``sources.csv`` and stored under ``data/raw``
- processed wastewater long-format data, if available in ``data/processed``

Predicted:
- UKHSA dashboard files classified as GP/admission series

Additional clinical datasets can be added once their raw columns are mapped to a
stable canonical time-series schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .io import load_sources, read_table, read_zip_tables
from .search_terms import (
    DEFAULT_SEARCH_TERMS_DIR,
    find_search_term_files,
    search_file_to_long,
)
from .ukhsa import build_ukhsa_series_catalogue, chart_to_series


@dataclass(frozen=True)
class SeriesSpec:
    """Metadata for one canonical time series."""

    series_id: str
    role: str
    dataset_family: str
    series_name: str
    source_file: str


def _coerce_series_frame(
    df: pd.DataFrame,
    *,
    spec: SeriesSpec,
    date_col: str = "date",
    value_col: str = "value",
) -> pd.DataFrame:
    """Return a canonical long-format time-series dataframe."""
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "value": pd.to_numeric(df[value_col], errors="coerce"),
            "series_id": spec.series_id,
            "role": spec.role,
            "dataset_family": spec.dataset_family,
            "series_name": spec.series_name,
            "source_file": spec.source_file,
        }
    ).dropna(subset=["date", "value"])
    return out


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw source column names for robust inference."""
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


def load_google_trends_1y_series(
    root: Path,
    search_dir: str | Path = DEFAULT_SEARCH_TERMS_DIR,
    value_column_index: int = 1,
) -> list[pd.DataFrame]:
    """Load one-year Google Trends files as predictive series."""
    files = find_search_term_files(root, search_dir=search_dir)
    series: list[pd.DataFrame] = []
    for row in files.to_dict(orient="records"):
        try:
            long = search_file_to_long(root, row["path"], value_column_index=value_column_index)
            term = str(long["search_term"].iloc[0]) if not long.empty else Path(row["path"]).stem
            spec = SeriesSpec(
                series_id=f"google_trends_1y::{term}",
                role="predictive",
                dataset_family="google_trends_1y",
                series_name=term,
                source_file=row["path"],
            )
            series.append(_coerce_series_frame(long, spec=spec, value_col="count"))
        except Exception:
            continue
    return series


def load_ukhsa_series(root: Path, series_type: str, role: str, dataset_family: str) -> list[pd.DataFrame]:
    """Load UKHSA chart files of one classified type."""
    catalogue = build_ukhsa_series_catalogue(root)
    if catalogue.empty or "series_type" not in catalogue.columns:
        return []

    series: list[pd.DataFrame] = []
    selected = catalogue[catalogue["series_type"] == series_type]
    for row in selected.to_dict(orient="records"):
        try:
            raw = chart_to_series(root, row["path"], series_type=series_type)
            name = Path(row["path"]).stem
            spec = SeriesSpec(
                series_id=f"{dataset_family}::{name}",
                role=role,
                dataset_family=dataset_family,
                series_name=name,
                source_file=row["path"],
            )
            series.append(_coerce_series_frame(raw, spec=spec, value_col="value"))
        except Exception:
            continue
    return series


def _infer_raw_date_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "date",
        "sample_date",
        "sampling_date",
        "collection_date",
        "week",
        "week_start",
        "week_ending",
        "datum",
        "date_start",
        "month",
        "period",
        "time",
    ]
    for col in candidates:
        if col in df.columns and pd.to_datetime(df[col], errors="coerce").notna().any():
            return col
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().sum() >= max(3, len(df) // 4):
            return col
    return None


def _raw_value_column_score(name: str) -> int:
    positive_terms = [
        "value",
        "viral",
        "virus",
        "load",
        "rna",
        "gene",
        "copies",
        "copy",
        "concentration",
        "gc",
        "mgc",
        "signal",
        "indicator",
        "average",
        "mean",
        "sars",
        "cov",
        "influenza",
        "flu",
        "rsv",
        "niv",
        "variant",
    ]
    negative_terms = [
        "id",
        "code",
        "date",
        "week",
        "month",
        "year",
        "day",
        "population",
        "pop",
        "lat",
        "lon",
        "long",
        "latitude",
        "longitude",
        "x",
        "y",
        "postcode",
        "postal",
    ]
    lower = name.lower()
    if any(term == lower or lower.endswith("_" + term) or lower.startswith(term + "_") for term in negative_terms):
        return -10
    return sum(term in lower for term in positive_terms)


def _infer_raw_value_columns(df: pd.DataFrame, date_col: str, max_columns: int = 3) -> list[str]:
    scored: list[tuple[int, str]] = []
    for col in df.columns:
        if col == date_col:
            continue
        values = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
        if values.notna().sum() < max(3, len(df) // 5):
            continue
        score = _raw_value_column_score(str(col))
        if score < 0:
            continue
        scored.append((score, str(col)))

    if not scored:
        return []
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    positive = [col for score, col in scored if score > 0]
    if positive:
        return positive[:max_columns]
    return [col for _, col in scored[:max_columns]]


def _raw_table_to_series(
    df: pd.DataFrame,
    *,
    country: str,
    pathogen_scope: str,
    level: str,
    source_file: str,
    table_name: str,
    max_value_columns: int = 3,
) -> list[pd.DataFrame]:
    """Convert one raw wastewater table into one or more predictive series."""
    if df is None or df.empty:
        return []
    table = _normalise_columns(df)
    date_col = _infer_raw_date_column(table)
    if date_col is None:
        return []
    value_cols = _infer_raw_value_columns(table, date_col=date_col, max_columns=max_value_columns)
    if not value_cols:
        return []

    series: list[pd.DataFrame] = []
    for value_col in value_cols:
        series_name = " / ".join(
            part for part in [country, pathogen_scope, level, table_name, value_col] if str(part).strip()
        )
        safe_name = series_name.replace("::", "_")
        spec = SeriesSpec(
            series_id=f"wastewater_raw::{safe_name}",
            role="predictive",
            dataset_family="wastewater_raw",
            series_name=series_name,
            source_file=source_file,
        )
        converted = pd.DataFrame(
            {
                "date": pd.to_datetime(table[date_col], errors="coerce"),
                "value": pd.to_numeric(table[value_col].astype(str).str.replace(",", "", regex=False), errors="coerce"),
            }
        ).dropna(subset=["date", "value"])
        if len(converted) >= 3:
            series.append(_coerce_series_frame(converted, spec=spec))
    return series


def _iter_raw_wastewater_tables(root: Path) -> Iterable[tuple[dict, str, pd.DataFrame]]:
    """Yield raw wastewater tables from sources.csv and data/raw.

    The downloader stores raw files under ``data/raw``. ZIP sources may also have
    extracted folders named after the ZIP stem, so both representations are
    supported.
    """
    root = Path(root)
    raw_dir = root / "data" / "raw"
    try:
        sources = load_sources(root)
    except Exception:
        return

    for row in sources.to_dict(orient="records"):
        filename = str(row.get("filename", "")).strip()
        if not filename:
            continue
        path = raw_dir / filename
        paths_to_try: list[Path] = []
        if path.exists() and path.is_file():
            paths_to_try.append(path)
        extracted_dir = raw_dir / Path(filename).stem
        if extracted_dir.exists() and extracted_dir.is_dir():
            paths_to_try.extend(
                sorted(p for p in extracted_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".json"})
            )

        for candidate in paths_to_try:
            try:
                if candidate.suffix.lower() == ".zip":
                    for member, table in read_zip_tables(candidate).items():
                        yield row, f"{candidate.name}:{member}", table
                elif candidate.suffix.lower() in {".csv", ".tsv", ".json"}:
                    yield row, candidate.relative_to(raw_dir).as_posix(), read_table(candidate)
            except Exception:
                continue


def load_raw_wastewater_series(root: Path, max_value_columns_per_table: int = 3) -> list[pd.DataFrame]:
    """Load raw wastewater files from ``data/raw`` as predictive series.

    This is intentionally heuristic because upstream wastewater schemas differ
    substantially by country. Tables with no parseable date/value columns are
    skipped and the rest are exposed as candidate predictive series.
    """
    series: list[pd.DataFrame] = []
    for row, table_name, table in _iter_raw_wastewater_tables(root):
        try:
            series.extend(
                _raw_table_to_series(
                    table,
                    country=str(row.get("country", "")),
                    pathogen_scope=str(row.get("pathogen_scope", "")),
                    level=str(row.get("level", "")),
                    source_file=str(row.get("filename", "")),
                    table_name=table_name,
                    max_value_columns=max_value_columns_per_table,
                )
            )
        except Exception:
            continue
    return series


def _read_processed_wastewater(root: Path) -> pd.DataFrame | None:
    processed = Path(root) / "data" / "processed"
    parquet = processed / "wastewater_long.parquet"
    csv = processed / "wastewater_long.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    return None


def load_processed_wastewater_series(root: Path) -> list[pd.DataFrame]:
    """Load processed wastewater long-format data as predictive series if present."""
    df = _read_processed_wastewater(root)
    if df is None or df.empty:
        return []

    if "date" not in df.columns:
        return []

    value_candidates = ["normalised_value", "zscore_within_series", "log10_value", "value"]
    value_col = next((col for col in value_candidates if col in df.columns), None)
    if value_col is None:
        return []

    group_cols = [col for col in ["country", "pathogen", "site_id", "region", "geography_level"] if col in df.columns]
    if not group_cols:
        group_cols = ["source_file"] if "source_file" in df.columns else []

    groups = [("wastewater", df)] if not group_cols else df.groupby(group_cols, dropna=False)
    series: list[pd.DataFrame] = []
    for key, group in groups:
        if not isinstance(key, tuple):
            key = (key,)
        parts = [str(x) for x in key if pd.notna(x)]
        name = " / ".join(parts) if parts else "wastewater"
        source_file = str(group["source_file"].iloc[0]) if "source_file" in group.columns and not group.empty else "data/processed/wastewater_long"
        spec = SeriesSpec(
            series_id=f"wastewater_processed::{name}",
            role="predictive",
            dataset_family="wastewater_processed",
            series_name=name,
            source_file=source_file,
        )
        series.append(_coerce_series_frame(group, spec=spec, value_col=value_col))
    return series


def build_available_series(root: Path) -> pd.DataFrame:
    """Build a canonical dataframe containing all automatically available series."""
    root = Path(root)
    parts: list[pd.DataFrame] = []
    parts.extend(load_google_trends_1y_series(root))
    parts.extend(load_ukhsa_series(root, series_type="nhs_calls", role="predictive", dataset_family="ukhsa_nhs_calls"))
    parts.extend(load_raw_wastewater_series(root))
    parts.extend(load_processed_wastewater_series(root))
    parts.extend(load_ukhsa_series(root, series_type="gp_admissions", role="predicted", dataset_family="ukhsa_gp_admissions"))

    if not parts:
        return pd.DataFrame(
            columns=["date", "value", "series_id", "role", "dataset_family", "series_name", "source_file"]
        )
    return pd.concat(parts, ignore_index=True)


def summarise_available_series(series: pd.DataFrame) -> pd.DataFrame:
    """Summarise available canonical series by role/family/id."""
    if series.empty:
        return pd.DataFrame()
    return (
        series.groupby(["role", "dataset_family", "series_id", "series_name", "source_file"], dropna=False)
        .agg(start_date=("date", "min"), end_date=("date", "max"), n_observations=("date", "size"))
        .reset_index()
        .sort_values(["role", "dataset_family", "series_name"])
    )


def expected_family_status(root: Path, series: pd.DataFrame) -> pd.DataFrame:
    """Return a small status table for expected predictive/predicted families."""
    rows = [
        {
            "role": "predictive",
            "dataset_family": "google_trends_1y",
            "expected_location": "Google_trends_v2/1y_data/time_series_GB*",
        },
        {
            "role": "predictive",
            "dataset_family": "ukhsa_nhs_calls",
            "expected_location": "ukhsa-chart* classified as nhs_calls",
        },
        {
            "role": "predictive",
            "dataset_family": "wastewater_raw",
            "expected_location": "sources.csv filenames under data/raw/",
        },
        {
            "role": "predictive",
            "dataset_family": "wastewater_processed",
            "expected_location": "optional data/processed/wastewater_long.{parquet,csv}",
        },
        {
            "role": "predicted",
            "dataset_family": "ukhsa_gp_admissions",
            "expected_location": "ukhsa-chart* classified as gp_admissions",
        },
    ]
    status = pd.DataFrame(rows)
    if series.empty:
        status["n_series_found"] = 0
    else:
        counts = series.groupby(["role", "dataset_family"])["series_id"].nunique().reset_index(name="n_series_found")
        status = status.merge(counts, on=["role", "dataset_family"], how="left")
        status["n_series_found"] = status["n_series_found"].fillna(0).astype(int)
    status["available"] = status["n_series_found"] > 0
    return status


def _period_column(date: pd.Series, freq: str) -> pd.Series:
    if freq.upper().startswith("M"):
        return pd.to_datetime(date).dt.to_period("M").dt.to_timestamp()
    return pd.to_datetime(date).dt.to_period("W").dt.start_time


def _aggregate_one_series(df: pd.DataFrame, freq: str = "W", aggregation: str = "mean") -> pd.DataFrame:
    out = df.copy()
    out["period"] = _period_column(out["date"], freq)
    if aggregation == "sum":
        grouped = out.groupby("period", dropna=False)["value"].sum(min_count=1)
    else:
        grouped = out.groupby("period", dropna=False)["value"].mean()
    return grouped.reset_index(name="value").dropna(subset=["period", "value"]).sort_values("period")


def _standardise_with_train(values: pd.Series, train_index: pd.Index) -> tuple[pd.Series, float, float]:
    train_values = pd.to_numeric(values.loc[train_index], errors="coerce")
    mean = float(train_values.mean())
    std = float(train_values.std())
    if not np.isfinite(std) or std == 0:
        std = 1.0
    return (pd.to_numeric(values, errors="coerce") - mean) / std, mean, std


def _metrics(y_true: pd.Series, y_pred: pd.Series, baseline_pred: float = 0.0) -> dict[str, float]:
    y_true = pd.Series(y_true, dtype="float64")
    y_pred = pd.Series(y_pred, dtype="float64")
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - np.sum(err**2) / denom) if denom > 0 else float("nan")
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    baseline_err = y_true - baseline_pred
    baseline_rmse = float(np.sqrt(np.mean(baseline_err**2)))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "correlation": corr,
        "baseline_rmse": baseline_rmse,
        "mse_skill_vs_train_mean": float(1 - (rmse**2 / baseline_rmse**2)) if baseline_rmse > 0 else float("nan"),
    }


def fit_pair_train_test(
    predictor: pd.DataFrame,
    target: pd.DataFrame,
    *,
    predictor_id: str,
    target_id: str,
    freq: str = "W",
    lags: Iterable[int] = (0, 1, 2, 3, 4),
    train_fraction: float = 0.8,
    min_test_size: int = 4,
    aggregation: str = "mean",
) -> tuple[dict[str, object], pd.DataFrame]:
    """Fit one predictive series against one predicted series with a chronological split."""
    x = _aggregate_one_series(predictor, freq=freq, aggregation=aggregation).rename(columns={"value": "x"})
    y = _aggregate_one_series(target, freq=freq, aggregation=aggregation).rename(columns={"value": "y"})
    frame = pd.merge(x, y, on="period", how="inner").sort_values("period")

    for lag in lags:
        frame[f"x_lag{lag}"] = frame["x"].shift(lag)
    predictor_cols = [f"x_lag{lag}" for lag in lags]
    model_df = frame[["period", "y", *predictor_cols]].dropna().copy()

    n = len(model_df)
    if n < 3:
        raise ValueError(f"Not enough overlapping complete observations for {predictor_id} -> {target_id}: {n}")

    if n <= min_test_size:
        split_idx = max(1, n - 1)
    else:
        split_idx = int(np.floor(n * train_fraction))
        split_idx = min(split_idx, n - min_test_size)
        split_idx = max(1, split_idx)

    train_idx = model_df.index[:split_idx]
    test_idx = model_df.index[split_idx:]

    y_z, y_mean, y_std = _standardise_with_train(model_df["y"], train_idx)
    model_df["y_z"] = y_z
    for col in predictor_cols:
        model_df[col + "_z"], _, _ = _standardise_with_train(model_df[col], train_idx)
    z_predictors = [col + "_z" for col in predictor_cols]

    train = model_df.loc[train_idx].copy()
    test = model_df.loc[test_idx].copy()
    X_train = sm.add_constant(train[z_predictors], has_constant="add")
    X_test = sm.add_constant(test[z_predictors], has_constant="add").reindex(columns=X_train.columns, fill_value=0.0)
    model = sm.OLS(train["y_z"], X_train).fit()

    train_pred = pd.Series(model.predict(X_train), index=train.index)
    test_pred = pd.Series(model.predict(X_test), index=test.index)
    metrics = _metrics(test["y_z"], test_pred, baseline_pred=0.0)

    prediction_rows = pd.concat(
        [
            train.assign(split="train", prediction=train_pred, residual=train["y_z"] - train_pred),
            test.assign(split="test", prediction=test_pred, residual=test["y_z"] - test_pred),
        ],
        axis=0,
    )
    prediction_rows["predictor_id"] = predictor_id
    prediction_rows["target_id"] = target_id
    prediction_rows["y_train_mean"] = y_mean
    prediction_rows["y_train_std"] = y_std

    result = {
        "predictor_id": predictor_id,
        "target_id": target_id,
        "freq": freq,
        "lags": ",".join(map(str, lags)),
        "n_complete": int(n),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_start": train["period"].min(),
        "train_end": train["period"].max(),
        "test_start": test["period"].min(),
        "test_end": test["period"].max(),
        **metrics,
    }
    return result, prediction_rows


def run_pairwise_train_test_matrix(
    series: pd.DataFrame,
    *,
    freq: str = "W",
    lags: Iterable[int] = (0, 1, 2, 3, 4),
    train_fraction: float = 0.8,
    min_test_size: int = 4,
    aggregation: str = "mean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every predictive series against every predicted series."""
    if series.empty:
        return pd.DataFrame(), pd.DataFrame()

    predictive_ids = sorted(series.loc[series["role"] == "predictive", "series_id"].dropna().unique())
    predicted_ids = sorted(series.loc[series["role"] == "predicted", "series_id"].dropna().unique())

    results: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    metadata = (
        series[["series_id", "role", "dataset_family", "series_name", "source_file"]]
        .drop_duplicates("series_id")
        .set_index("series_id")
    )

    for predictor_id in predictive_ids:
        predictor = series[series["series_id"] == predictor_id]
        for target_id in predicted_ids:
            target = series[series["series_id"] == target_id]
            try:
                result, pred = fit_pair_train_test(
                    predictor,
                    target,
                    predictor_id=predictor_id,
                    target_id=target_id,
                    freq=freq,
                    lags=lags,
                    train_fraction=train_fraction,
                    min_test_size=min_test_size,
                    aggregation=aggregation,
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
    if not results_df.empty and "mse_skill_vs_train_mean" in results_df.columns:
        results_df = results_df.sort_values("mse_skill_vs_train_mean", ascending=False, na_position="last")

    predictions_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return results_df, predictions_df

"""Regression helpers for search-term time-series files.

Search-term files are expected to have filenames beginning with ``time_series_GB``.
The search-volume predictor is always the second source column by position,
regardless of the column name. These helpers scan a local checkout, load the
files, coerce that second column to a canonical ``count`` field, and build
regression frames against UKHSA GP/admission chart series.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .ukhsa import build_ukhsa_series_catalogue, chart_to_series


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


def find_search_term_files(root: Path, pattern: str = "time_series_GB*") -> pd.DataFrame:
    """Find search-term files anywhere in the repository checkout."""
    root = Path(root).resolve()
    files = sorted(
        p for p in root.rglob(pattern)
        if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".txt", ".json"}
    )
    return pd.DataFrame(
        {
            "path": [p.relative_to(root).as_posix() for p in files],
            "filename": [p.name for p in files],
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


def build_search_term_catalogue(root: Path, value_column_index: int = 1) -> pd.DataFrame:
    """Return file metadata plus inferred schema for every search-term file."""
    files = find_search_term_files(root)
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


def build_search_term_regression_frame(
    root: Path,
    search_files: Sequence[str] | None = None,
    outcome_files: Sequence[str] | None = None,
    freq: str = "W",
    lags: Iterable[int] = (0, 1, 2, 3, 4),
    aggregate_terms: bool = True,
    value_column_index: int = 1,
) -> pd.DataFrame:
    """Build a lagged regression frame for search-term counts vs GP admissions.

    Parameters
    ----------
    aggregate_terms:
        If True, all second-column values are summed into ``search_count`` before
        lagging. If False, each search term is pivoted into separate predictors,
        still using the second source column from each file.
    value_column_index:
        Zero-based source-column index to use as the search-volume predictor.
        The default, 1, means the second column.
    """
    root = Path(root)
    if search_files is None:
        search_files = find_search_term_files(root)["path"].tolist()
    if not search_files:
        raise ValueError("No search-term files matching time_series_GB* were found.")

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
    for col in [c for c in frame.columns if c not in {"period", "gp_admissions", "z_gp_admissions"}]:
        zcol = "z_" + re.sub(r"[^0-9A-Za-z]+", "_", str(col)).strip("_").lower()
        frame[zcol] = standardise(frame[col])
        for lag in lags:
            frame[f"{zcol}_lag{lag}"] = frame[zcol].shift(lag)
    return frame


def fit_search_term_ols(
    frame: pd.DataFrame,
    lags: Iterable[int] = (0, 1, 2, 3, 4),
    predictor_prefix: str = "z_search_count_lag",
    seasonal_controls: bool = True,
):
    """Fit OLS for z-scored GP admissions on lagged search-term counts."""
    predictors = [f"{predictor_prefix}{lag}" for lag in lags]
    model_df = frame[["period", "z_gp_admissions", *predictors]].dropna().copy()
    y = model_df["z_gp_admissions"]
    X = model_df[predictors].copy()
    if seasonal_controls:
        month = pd.to_datetime(model_df["period"]).dt.month.astype("category")
        X = pd.concat([X, pd.get_dummies(month, prefix="month", drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(lags) if lags else 1})

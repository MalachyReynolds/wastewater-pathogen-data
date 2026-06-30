"""Helpers for NHS111 / clinical activity regression analyses."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


def normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
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


def read_clinical_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read an NHS England CSV with tolerant encoding and column cleanup."""
    path = Path(path)
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1", **kwargs)
    return normalise_column_names(df)


def list_clinical_files(root: Path, domain: str | None = None) -> pd.DataFrame:
    """Inventory downloaded clinical CSV files."""
    raw = Path(root) / "data" / "clinical" / "raw"
    files = sorted(raw.rglob("*.csv")) if raw.exists() else []
    if domain:
        files = [p for p in files if domain in p.parts]
    return pd.DataFrame(
        {
            "path": [p.relative_to(root).as_posix() for p in files],
            "domain": [p.relative_to(raw).parts[0] if len(p.relative_to(raw).parts) > 1 else "" for p in files],
            "year": [p.relative_to(raw).parts[1] if len(p.relative_to(raw).parts) > 2 else "" for p in files],
            "size_kb": [p.stat().st_size / 1024 for p in files],
        }
    )


def find_columns(df: pd.DataFrame, include: Sequence[str], exclude: Sequence[str] = ()) -> list[str]:
    """Find columns whose normalised names include all terms and no excludes."""
    include = [x.lower() for x in include]
    exclude = [x.lower() for x in exclude]
    matches = []
    for col in df.columns:
        name = str(col).lower()
        if all(term in name for term in include) and not any(term in name for term in exclude):
            matches.append(col)
    return matches


def coerce_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce counts stored with commas/suppression symbols into numbers."""
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("*", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def infer_month_from_columns(df: pd.DataFrame) -> pd.Series | None:
    """Try to infer a monthly date column from common NHS column names."""
    candidates = [
        "month",
        "period",
        "reporting_period",
        "reporting_month",
        "date",
        "month_start",
    ]
    for col in candidates:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                return parsed.dt.to_period("M").dt.to_timestamp()
    return None


def aggregate_file_to_monthly(
    path: Path,
    value_column: str,
    date_column: str | None = None,
    group_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Aggregate one source file to monthly totals for a selected value column."""
    df = read_clinical_csv(path)
    if value_column not in df.columns:
        raise KeyError(f"{value_column!r} not found in {path}; available columns={list(df.columns)}")

    if date_column:
        month = pd.to_datetime(df[date_column], errors="coerce").dt.to_period("M").dt.to_timestamp()
    else:
        inferred = infer_month_from_columns(df)
        if inferred is None:
            # Many NHS monthly CSVs are one month per file. Use the parent year plus filename where possible.
            inferred = pd.Series(pd.NaT, index=df.index)
        month = inferred

    work = pd.DataFrame({"month": month, "value": coerce_numeric_series(df[value_column])})
    for col in group_cols:
        if col in df.columns:
            work[col] = df[col]
    group_by = ["month", *[c for c in group_cols if c in work.columns]]
    return work.groupby(group_by, dropna=False)["value"].sum(min_count=1).reset_index()


def create_lags(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    lags: Iterable[int],
    time_col: str = "month",
    group_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Create lagged columns for time series regression."""
    out = df.sort_values([*group_cols, time_col]).copy() if group_cols else df.sort_values(time_col).copy()
    grouped = out.groupby(list(group_cols), dropna=False) if group_cols else [(None, out)]

    if group_cols:
        for col in value_cols:
            for lag in lags:
                out[f"{col}_lag{lag}"] = out.groupby(list(group_cols), dropna=False)[col].shift(lag)
    else:
        for col in value_cols:
            for lag in lags:
                out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def fit_ols(
    df: pd.DataFrame,
    outcome: str,
    predictors: Sequence[str],
    add_month_dummies: bool = True,
    time_col: str = "month",
):
    """Fit an OLS regression with optional month-of-year controls and HAC errors."""
    model_df = df[[outcome, *predictors, time_col]].dropna().copy()
    y = model_df[outcome]
    X = model_df[list(predictors)].copy()

    if add_month_dummies:
        month_num = pd.to_datetime(model_df[time_col]).dt.month.astype("category")
        dummies = pd.get_dummies(month_num, prefix="calendar_month", drop_first=True, dtype=float)
        X = pd.concat([X, dummies], axis=1)

    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X)
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": 3})


def standardise_series(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Add z-scored versions of selected columns."""
    out = df.copy()
    for col in cols:
        values = pd.to_numeric(out[col], errors="coerce")
        std = values.std(skipna=True)
        out[f"z_{col}"] = (values - values.mean(skipna=True)) / std if std else np.nan
    return out


def quick_column_report(path: Path, max_cols: int = 80) -> dict:
    df = read_clinical_csv(path, nrows=50)
    return {
        "path": str(path),
        "n_preview_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns[:max_cols]),
        "call_like_columns": find_columns(df, ["call"]),
        "admission_like_columns": find_columns(df, ["admission"]),
        "gp_like_columns": find_columns(df, ["gp"]),
    }

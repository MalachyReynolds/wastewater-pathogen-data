"""Helpers for UKHSA dashboard chart CSV files.

The UKHSA dashboard export files in this repository have filenames beginning
with ``ukhsa-chart``. These helpers scan the local checkout, load those files,
infer sensible date/value columns, and build a regression frame for NHS calls
against GP/admission outcomes.
"""
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


def find_ukhsa_chart_files(root: Path, pattern: str = "ukhsa-chart*") -> pd.DataFrame:
    """Find UKHSA chart files anywhere in the repository checkout."""
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


def _read_delimited(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    # UKHSA chart exports are usually CSVs; use a light fallback for semicolon files.
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")
    except Exception:
        return pd.read_csv(path, sep=";")


def read_ukhsa_chart(path: Path) -> pd.DataFrame:
    """Read one UKHSA chart export and normalise column names."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        df = pd.read_json(path)
    else:
        df = _read_delimited(path)
    return normalise_column_names(df)


def find_date_column(df: pd.DataFrame) -> str:
    candidates = [
        "date",
        "week",
        "epiweek",
        "epi_week",
        "week_start",
        "week_ending",
        "month",
        "period",
        "x",
    ]
    for col in candidates:
        if col in df.columns and pd.to_datetime(df[col], errors="coerce").notna().any():
            return col
    for col in df.columns:
        if pd.to_datetime(df[col], errors="coerce").notna().sum() >= max(3, len(df) // 3):
            return col
    raise ValueError(f"Could not infer date column from columns={list(df.columns)}")


def numeric_columns(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        values = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
        if values.notna().sum() >= max(3, len(values) // 3):
            out.append(col)
    return out


def find_value_column(df: pd.DataFrame, preferred_terms: Sequence[str] = ()) -> str:
    nums = numeric_columns(df)
    if not nums:
        raise ValueError(f"Could not infer numeric value column from columns={list(df.columns)}")

    terms = [t.lower() for t in preferred_terms]
    if terms:
        scored = []
        for col in nums:
            name = col.lower()
            score = sum(term in name for term in terms)
            scored.append((score, col))
        scored.sort(reverse=True)
        if scored[0][0] > 0:
            return scored[0][1]

    # Avoid using obvious date/time columns as values.
    for col in nums:
        if not any(term in col.lower() for term in ["date", "week", "month", "year"]):
            return col
    return nums[0]


def classify_ukhsa_chart(path: Path, df: pd.DataFrame | None = None) -> str:
    """Classify a UKHSA chart as predictor/outcome/other using filename and columns."""
    path = Path(path)
    text = path.name.lower().replace("_", "-")
    if df is not None:
        text += " " + " ".join(map(str, df.columns)).lower()

    nhs_terms = ["nhs111", "nhs-111", "nhs_111", "111", "call", "calls", "online"]
    gp_terms = ["gp", "general-practice", "general_practice", "admission", "admissions", "consultation", "consultations"]

    has_nhs = any(term in text for term in nhs_terms)
    has_gp = any(term in text for term in gp_terms)

    if has_nhs and not has_gp:
        return "nhs_calls"
    if has_gp and not has_nhs:
        return "gp_admissions"
    if has_nhs and has_gp:
        # Use more specific GP/admission terms to break ties.
        if "admission" in text or "gp" in text:
            return "gp_admissions"
        return "nhs_calls"
    return "other"


def chart_to_series(
    root: Path,
    rel_path: str,
    series_type: str | None = None,
    value_terms: Sequence[str] = ("value", "count", "number", "calls", "admissions"),
) -> pd.DataFrame:
    """Convert one UKHSA chart file to a canonical time series."""
    path = Path(root) / rel_path
    df = read_ukhsa_chart(path)
    date_col = find_date_column(df)
    value_col = find_value_column(df, preferred_terms=value_terms)
    series_type = series_type or classify_ukhsa_chart(path, df)

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "value": pd.to_numeric(df[value_col].astype(str).str.replace(",", "", regex=False), errors="coerce"),
            "series_type": series_type,
            "source_file": rel_path,
            "date_column": date_col,
            "value_column": value_col,
        }
    ).dropna(subset=["date", "value"])

    out["week"] = out["date"].dt.to_period("W").dt.start_time
    out["month"] = out["date"].dt.to_period("M").dt.to_timestamp()
    return out


def build_ukhsa_series_catalogue(root: Path) -> pd.DataFrame:
    """Return UKHSA chart file metadata plus inferred type/date/value columns."""
    files = find_ukhsa_chart_files(root)
    rows = []
    for row in files.to_dict(orient="records"):
        path = Path(root) / row["path"]
        try:
            df = read_ukhsa_chart(path)
            date_col = find_date_column(df)
            value_col = find_value_column(df, preferred_terms=("value", "count", "number", "calls", "admissions"))
            series_type = classify_ukhsa_chart(path, df)
            rows.append({**row, "series_type": series_type, "date_column": date_col, "value_column": value_col, "columns": list(df.columns)})
        except Exception as exc:
            rows.append({**row, "series_type": "error", "date_column": "", "value_column": "", "columns": [], "error": repr(exc)})
    return pd.DataFrame(rows)


def standardise(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    std = values.std(skipna=True)
    if pd.isna(std) or std == 0:
        return values * np.nan
    return (values - values.mean(skipna=True)) / std


def build_regression_frame_from_ukhsa(
    root: Path,
    predictor_files: Sequence[str] | None = None,
    outcome_files: Sequence[str] | None = None,
    freq: str = "W",
    lags: Iterable[int] = (0, 1, 2, 3, 4),
) -> pd.DataFrame:
    """Build a regression frame using UKHSA NHS calls and GP/admission chart files.

    Parameters
    ----------
    predictor_files:
        UKHSA chart files for NHS 111/call activity. If omitted, files inferred
        as ``nhs_calls`` are used.
    outcome_files:
        UKHSA chart files for GP/admission activity. If omitted, files inferred
        as ``gp_admissions`` are used.
    freq:
        ``"W"`` for weekly aggregation or ``"M"`` for monthly aggregation.
    lags:
        Lags in periods of ``freq`` for the NHS calls predictor.
    """
    root = Path(root)
    catalogue = build_ukhsa_series_catalogue(root)

    if predictor_files is None:
        predictor_files = catalogue.loc[catalogue["series_type"] == "nhs_calls", "path"].tolist()
    if outcome_files is None:
        outcome_files = catalogue.loc[catalogue["series_type"] == "gp_admissions", "path"].tolist()

    if not predictor_files:
        raise ValueError("No NHS-call UKHSA chart files found. Pass predictor_files explicitly.")
    if not outcome_files:
        raise ValueError("No GP/admission UKHSA chart files found. Pass outcome_files explicitly.")

    predictor = pd.concat([chart_to_series(root, p, series_type="nhs_calls") for p in predictor_files], ignore_index=True)
    outcome = pd.concat([chart_to_series(root, p, series_type="gp_admissions") for p in outcome_files], ignore_index=True)

    period_col = "week" if freq.upper().startswith("W") else "month"
    x = predictor.groupby(period_col, dropna=False)["value"].sum(min_count=1).reset_index(name="nhs_calls")
    y = outcome.groupby(period_col, dropna=False)["value"].sum(min_count=1).reset_index(name="gp_admissions")

    out = pd.merge(x, y, on=period_col, how="inner").rename(columns={period_col: "period"}).sort_values("period")
    out["z_nhs_calls"] = standardise(out["nhs_calls"])
    out["z_gp_admissions"] = standardise(out["gp_admissions"])

    for lag in lags:
        out[f"z_nhs_calls_lag{lag}"] = out["z_nhs_calls"].shift(lag)
    return out


def fit_ukhsa_lagged_ols(
    frame: pd.DataFrame,
    lags: Iterable[int] = (0, 1, 2, 3, 4),
    seasonal_controls: bool = True,
):
    """Fit a lagged OLS model for GP/admissions on NHS-call predictors."""
    predictors = [f"z_nhs_calls_lag{lag}" for lag in lags]
    model_df = frame[["period", "z_gp_admissions", *predictors]].dropna().copy()
    y = model_df["z_gp_admissions"]
    X = model_df[predictors].copy()

    if seasonal_controls:
        month = pd.to_datetime(model_df["period"]).dt.month.astype("category")
        X = pd.concat([X, pd.get_dummies(month, prefix="month", drop_first=True, dtype=float)], axis=1)

    X = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(lags) if lags else 1})

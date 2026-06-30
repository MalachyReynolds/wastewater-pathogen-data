"""Cleaning and feature engineering helpers for wastewater analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with simple snake_case-ish column names."""
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


def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add date/week/year/month columns from a date column."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["date"] = out[date_col]
    out["week"] = out["date"].dt.to_period("W").dt.start_time
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    return out


def add_log_signal(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Add a log10 signal column, ignoring non-positive values."""
    out = df.copy()
    values = pd.to_numeric(out[value_col], errors="coerce")
    out["log10_value"] = np.log10(values.where(values > 0))
    return out


def add_within_series_zscore(
    df: pd.DataFrame,
    value_col: str = "log10_value",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add within-series z-scores for cross-site/cross-country comparison.

    Use this for relative comparisons. Do not compare raw absolute viral loads
    across countries without checking units and laboratory methods.
    """
    out = df.copy()
    group_cols = group_cols or ["country", "pathogen", "site_id"]

    def zscore(x: pd.Series) -> pd.Series:
        std = x.std(skipna=True)
        if pd.isna(std) or std == 0:
            return x * np.nan
        return (x - x.mean(skipna=True)) / std

    out["zscore_within_series"] = out.groupby(group_cols, dropna=False)[value_col].transform(zscore)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    value_col: str = "log10_value",
    group_cols: list[str] | None = None,
    window: int = 3,
) -> pd.DataFrame:
    """Add rolling mean and first-difference trend features."""
    out = df.copy().sort_values("date")
    group_cols = group_cols or ["country", "pathogen", "site_id"]
    rolling_col = f"rolling_{window}_{value_col}"
    change_col = f"weekly_change_{rolling_col}"

    out[rolling_col] = (
        out.groupby(group_cols, dropna=False)[value_col]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    out[change_col] = out.groupby(group_cols, dropna=False)[rolling_col].diff()
    return out


def canonical_empty_frame() -> pd.DataFrame:
    """Return an empty canonical long-format wastewater dataframe."""
    return pd.DataFrame(
        columns=[
            "country",
            "source",
            "pathogen",
            "date",
            "site_id",
            "site_name",
            "geography_level",
            "region",
            "value",
            "value_unit",
            "population",
            "normalised_value",
            "normalised_unit",
            "source_file",
        ]
    )


def latest_signal_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise latest z-scored signal by country and pathogen."""
    latest = df.sort_values("date").groupby(["country", "pathogen", "site_id"], dropna=False).tail(1)
    return (
        latest.groupby(["country", "pathogen"], dropna=False)
        .agg(
            latest_date=("date", "max"),
            sites=("site_id", "nunique"),
            median_latest_z=("zscore_within_series", "median"),
        )
        .reset_index()
        .sort_values(["pathogen", "median_latest_z"], ascending=[True, False])
    )

"""External public respiratory series loaders.

These helpers turn downloaded external files into the same canonical long format
used by the repository's modelling code. They are deliberately conservative and
only read files that already exist under ``data/external``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .regression_matrix import build_available_series

CANONICAL_COLUMNS = ["date", "value", "series_id", "role", "dataset_family", "series_name", "source_file"]


def _canonical(
    df: pd.DataFrame,
    *,
    date_col: str,
    value_col: str,
    series_id: str,
    role: str,
    dataset_family: str,
    series_name: str,
    source_file: str,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "value": pd.to_numeric(df[value_col], errors="coerce"),
            "series_id": series_id,
            "role": role,
            "dataset_family": dataset_family,
            "series_name": series_name,
            "source_file": source_file,
        }
    ).dropna(subset=["date", "value"])
    return out[CANONICAL_COLUMNS]


def load_owid_covid_series(root: Path, location: str = "United Kingdom") -> list[pd.DataFrame]:
    """Load OWID COVID data as both lagged predictors and predicted targets.

    Predictive OWID series are only used through lags in ``ml_panel``. They
    therefore provide autoregressive COVID and surveillance context without using
    same-week target information.
    """
    path = Path(root) / "data" / "external" / "owid_covid_data.csv"
    if not path.exists():
        return []

    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if "location" in df.columns:
        df = df[df["location"] == location].copy()
    if df.empty or "date" not in df.columns:
        return []

    predictive_cols = [
        "new_cases_smoothed",
        "new_deaths_smoothed",
        "reproduction_rate",
        "new_tests_smoothed",
        "positive_rate",
        "people_vaccinated_per_hundred",
        "people_fully_vaccinated_per_hundred",
        "total_boosters_per_hundred",
        "new_vaccinations_smoothed_per_million",
        "stringency_index",
    ]
    predicted_cols = [
        "new_cases_smoothed",
        "weekly_hosp_admissions",
        "hosp_patients",
        "icu_patients",
        "new_deaths_smoothed",
    ]

    series: list[pd.DataFrame] = []
    source_file = path.relative_to(root).as_posix()
    for col in predictive_cols:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() >= 10:
            series.append(
                _canonical(
                    df,
                    date_col="date",
                    value_col=col,
                    series_id=f"owid_covid_predictive::{location}::{col}",
                    role="predictive",
                    dataset_family="owid_covid_predictive",
                    series_name=f"{location} {col}",
                    source_file=source_file,
                )
            )
    for col in predicted_cols:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() >= 10:
            series.append(
                _canonical(
                    df,
                    date_col="date",
                    value_col=col,
                    series_id=f"owid_covid_target::{location}::{col}",
                    role="predicted",
                    dataset_family="owid_covid_target",
                    series_name=f"{location} {col}",
                    source_file=source_file,
                )
            )
    return series


def _read_open_meteo_csv(path: Path) -> pd.DataFrame | None:
    """Read Open-Meteo CSV output, tolerating metadata preambles."""
    for skiprows in (0, 2, 3, 4, 5):
        try:
            df = pd.read_csv(path, skiprows=skiprows)
            if "time" in df.columns or "date" in df.columns:
                return df
        except Exception:
            continue
    return None


def load_open_meteo_weather_series(root: Path) -> list[pd.DataFrame]:
    """Load Open-Meteo weather files as predictive series."""
    weather_dir = Path(root) / "data" / "external" / "weather"
    if not weather_dir.exists():
        return []

    series: list[pd.DataFrame] = []
    for path in sorted(weather_dir.glob("open_meteo_*.csv")):
        df = _read_open_meteo_csv(path)
        if df is None or df.empty:
            continue
        date_col = "time" if "time" in df.columns else "date"
        geography = path.stem.replace("open_meteo_", "")
        source_file = path.relative_to(root).as_posix()
        for col in df.columns:
            if col == date_col:
                continue
            if pd.to_numeric(df[col], errors="coerce").notna().sum() < 10:
                continue
            series.append(
                _canonical(
                    df,
                    date_col=date_col,
                    value_col=col,
                    series_id=f"open_meteo_weather::{geography}::{col}",
                    role="predictive",
                    dataset_family="open_meteo_weather",
                    series_name=f"{geography} {col}",
                    source_file=source_file,
                )
            )
    return series


def build_all_available_series(root: Path, include_external: bool = True) -> pd.DataFrame:
    """Load local repository series plus downloaded external public series."""
    root = Path(root)
    parts: list[pd.DataFrame] = []
    local = build_available_series(root)
    if not local.empty:
        parts.append(local[CANONICAL_COLUMNS])
    if include_external:
        parts.extend(load_owid_covid_series(root))
        parts.extend(load_open_meteo_weather_series(root))
    if not parts:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    return pd.concat(parts, ignore_index=True)

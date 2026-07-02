from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..external_series import CANONICAL_COLUMNS, build_all_available_series

ALLOWED_FILE_SUFFIXES = {".csv", ".parquet", ".xlsx", ".xls"}
EXCLUDED_DIRS = {".git", ".venv", "__pycache__", ".ipynb_checkpoints"}


def discover_local_files(root: Path, limit: int = 200) -> list[Path]:
    """Discover likely tabular dataset files inside the workspace."""
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in ALLOWED_FILE_SUFFIXES:
            candidates.append(path)

    candidates = sorted({path.resolve() for path in candidates})
    return candidates[:limit]


def load_uploaded_or_local_file(uploaded_file=None, path: Path | None = None) -> pd.DataFrame:
    """Load a raw dataset file, either an uploaded Streamlit file or a local path."""
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file)

    if path is None:
        raise ValueError("Either uploaded_file or path must be provided.")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_canonical_series(root: Path, include_external: bool = True) -> pd.DataFrame:
    """Load the canonical long-format panel of every locally discoverable series."""
    return build_all_available_series(Path(root), include_external=include_external)


def summarise_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.describe(include="all").transpose()
    summary["missing"] = frame.isna().sum()
    summary["missing_pct"] = (summary["missing"] / len(frame) * 100).round(2)
    return summary


def list_series_catalogue(series: pd.DataFrame) -> pd.DataFrame:
    """Summarise the canonical series panel into one row per series_id."""
    if series.empty:
        return pd.DataFrame(
            columns=["series_id", "role", "dataset_family", "series_name", "n_obs", "date_min", "date_max"]
        )

    grouped = series.groupby("series_id", dropna=False).agg(
        role=("role", "first"),
        dataset_family=("dataset_family", "first"),
        series_name=("series_name", "first"),
        n_obs=("value", "count"),
        date_min=("date", "min"),
        date_max=("date", "max"),
    )
    return grouped.reset_index().sort_values(["role", "dataset_family", "series_id"]).reset_index(drop=True)


def build_custom_series(
    frame: pd.DataFrame,
    date_column: str,
    value_column: str,
    series_id: str,
    series_name: str,
    role: str,
    source_file: str = "uploaded",
) -> pd.DataFrame:
    """Convert an arbitrary raw dataframe column pair into one canonical-format series."""
    if role not in {"predictive", "predicted"}:
        raise ValueError("role must be 'predictive' or 'predicted'")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_column], errors="coerce"),
            "value": pd.to_numeric(frame[value_column], errors="coerce"),
            "series_id": series_id,
            "role": role,
            "dataset_family": "custom",
            "series_name": series_name,
            "source_file": source_file,
        }
    ).dropna(subset=["date", "value"])

    if out.empty:
        raise ValueError("No valid date/value rows were found in the chosen columns.")

    return out[CANONICAL_COLUMNS].sort_values("date").reset_index(drop=True)


def merge_series(series: pd.DataFrame | None, new_series: pd.DataFrame) -> pd.DataFrame:
    """Add a series to the canonical panel, replacing any existing series with the same id."""
    if series is None or series.empty:
        return new_series.reset_index(drop=True)

    series_id = new_series["series_id"].iloc[0]
    remaining = series[series["series_id"] != series_id]
    return pd.concat([remaining, new_series], ignore_index=True)

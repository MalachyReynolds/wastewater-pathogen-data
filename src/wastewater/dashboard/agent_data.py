"""Read agent-generated data artifacts for the Streamlit dashboard.

The autonomous data agent writes Parquet files plus JSON manifests. These helpers
adapt those artifacts to the dashboard's canonical long-format series panel.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..external_series import CANONICAL_COLUMNS

LATEST_DIR = Path("data_registry") / "latest"
NORMALIZED_DIR = Path("data") / "normalized"
DATE_CANDIDATES = ("date", "period", "week", "epi_week_start", "target_date")
ID_COLUMNS = {
    "date", "period", "week", "epi_year", "epi_week", "source", "pathogen",
    "signal_type", "metric", "unit", "geography_name", "geography_code",
    "geography_level", "age_group", "sex", "retrieved_at", "run_id", "feature_set",
}


def _safe_token(value: Any) -> str:
    text = str(value) if value is not None and str(value) else "unknown"
    text = re.sub(r"[^a-z0-9]+", "_", text.strip().lower())
    return text.strip("_") or "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _display(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _manifest_from_pointer(root: Path, pointer_path: Path) -> dict[str, Any] | None:
    try:
        pointer = _read_json(pointer_path)
    except Exception:
        return None
    if "manifest_path" in pointer:
        manifest_path = _resolve(root, pointer["manifest_path"])
        if not manifest_path.exists():
            return {
                "feature_set": pointer.get("feature_set", pointer_path.stem),
                "run_id": pointer.get("latest_run_id"),
                "status": "missing_manifest",
                "manifest_path": _display(root, manifest_path),
            }
        try:
            manifest = _read_json(manifest_path)
        except Exception:
            return None
        manifest.setdefault("feature_set", pointer.get("feature_set", pointer_path.stem))
        manifest.setdefault("run_id", pointer.get("latest_run_id"))
        manifest["manifest_path"] = _display(root, manifest_path)
        return manifest
    pointer.setdefault("feature_set", pointer_path.stem)
    pointer["manifest_path"] = _display(root, pointer_path)
    return pointer


def list_latest_agent_manifests(root: Path) -> pd.DataFrame:
    """Summarise the latest agent artifact manifests."""
    root = Path(root)
    latest_dir = root / LATEST_DIR
    rows: list[dict[str, Any]] = []
    if not latest_dir.exists():
        return pd.DataFrame(columns=["feature_set", "artifact_type", "run_id", "rows", "columns", "date_min", "date_max", "validation_status", "path", "manifest_path"])
    for pointer_path in sorted(latest_dir.glob("*.json")):
        manifest = _manifest_from_pointer(root, pointer_path)
        if manifest is None:
            continue
        rows.append(
            {
                "feature_set": manifest.get("feature_set", pointer_path.stem),
                "artifact_type": manifest.get("artifact_type", "unknown"),
                "run_id": manifest.get("run_id") or manifest.get("latest_run_id"),
                "rows": manifest.get("rows"),
                "columns": manifest.get("columns"),
                "date_min": manifest.get("date_min") or manifest.get("data_start"),
                "date_max": manifest.get("date_max") or manifest.get("data_end"),
                "validation_status": manifest.get("validation_status") or manifest.get("status"),
                "path": manifest.get("path"),
                "manifest_path": manifest.get("manifest_path"),
            }
        )
    return pd.DataFrame(rows)


def load_feature_table(root: Path, feature_set: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a feature table through its latest manifest pointer."""
    root = Path(root)
    pointer_path = root / LATEST_DIR / f"{feature_set}.json"
    if not pointer_path.exists():
        raise FileNotFoundError(f"No latest pointer found for feature set: {feature_set}")
    manifest = _manifest_from_pointer(root, pointer_path)
    if manifest is None:
        raise ValueError(f"Could not read latest manifest pointer: {pointer_path}")
    if "path" not in manifest:
        raise ValueError(f"Manifest for {feature_set} does not contain a data path.")
    path = _resolve(root, manifest["path"])
    if not path.exists():
        raise FileNotFoundError(f"Feature table does not exist: {_display(root, path)}")
    return pd.read_parquet(path), manifest


def _find_date_column(frame: pd.DataFrame) -> str:
    lowered = {column.lower(): column for column in frame.columns}
    for candidate in DATE_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            return column
    raise ValueError("Could not infer a date/period column for this table.")


def _feature_role(column: str) -> str:
    token = column.lower()
    if token.startswith("target") or "target_" in token:
        return "predicted"
    return "predictive"


def feature_table_to_canonical_series(frame: pd.DataFrame, *, feature_set: str, source_file: str, min_observations: int = 3) -> pd.DataFrame:
    """Convert a wide feature table to the dashboard canonical series format."""
    if frame.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    date_col = _find_date_column(frame)
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    parts: list[pd.DataFrame] = []
    for column in frame.columns:
        if column == date_col or column.lower() in ID_COLUMNS:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() < min_observations:
            continue
        out = pd.DataFrame(
            {
                "date": dates,
                "value": values,
                "series_id": f"agent_feature::{_safe_token(feature_set)}::{_safe_token(column)}",
                "role": _feature_role(column),
                "dataset_family": f"agent_feature::{feature_set}",
                "series_name": f"{feature_set}: {column}",
                "source_file": source_file,
            }
        ).dropna(subset=["date", "value"])
        if not out.empty:
            parts.append(out[CANONICAL_COLUMNS])
    if not parts:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    return pd.concat(parts, ignore_index=True).sort_values(["series_id", "date"]).reset_index(drop=True)


def load_normalized_signal_tables(root: Path) -> pd.DataFrame:
    """Load normalized agent signal Parquet files from data/normalized."""
    root = Path(root)
    base = root / NORMALIZED_DIR
    if not base.exists():
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for path in sorted(base.glob("**/*.parquet")):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        if not frame.empty:
            frame = frame.copy()
            frame["_source_file"] = _display(root, path)
            parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def normalized_signals_to_canonical_series(frame: pd.DataFrame, min_observations: int = 3) -> pd.DataFrame:
    """Convert normalized respiratory signal rows into canonical dashboard series."""
    if frame.empty or not {"date", "value"}.issubset(frame.columns):
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    work = frame.copy()
    for column in ["source", "pathogen", "signal_type", "metric", "geography_name", "geography_code"]:
        if column not in work.columns:
            work[column] = "unknown"
    if "_source_file" not in work.columns:
        work["_source_file"] = "data/normalized"
    has_explicit_role = "role" in work.columns
    parts: list[pd.DataFrame] = []
    keys = ["source", "pathogen", "signal_type", "metric", "geography_name", "geography_code"]
    for values_tuple, group in work.groupby(keys, dropna=False):
        numeric = pd.to_numeric(group["value"], errors="coerce")
        if numeric.notna().sum() < min_observations:
            continue
        source, pathogen, signal_type, metric, geography_name, geography_code = values_tuple
        # Prefer the role the source was explicitly classified with at add-time (written by
        # ingest.py) over guessing from the signal name -- the heuristic only exists for
        # normalized tables written before that classification existed.
        explicit_role = group["role"].iloc[0] if has_explicit_role else None
        if explicit_role in {"predictive", "predicted"}:
            role = explicit_role
        else:
            role = "predicted" if "admission" in f"{signal_type} {metric}".lower() else "predictive"
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(group["date"], errors="coerce"),
                "value": numeric,
                "series_id": "agent_signal::" + "::".join(_safe_token(value) for value in values_tuple),
                "role": role,
                "dataset_family": "agent_normalized_signal",
                "series_name": f"{geography_name} {pathogen} {signal_type} {metric}",
                "source_file": str(group["_source_file"].iloc[0]),
            }
        ).dropna(subset=["date", "value"])
        if not out.empty:
            parts.append(out[CANONICAL_COLUMNS])
    return pd.concat(parts, ignore_index=True).sort_values(["series_id", "date"]).reset_index(drop=True) if parts else pd.DataFrame(columns=CANONICAL_COLUMNS)

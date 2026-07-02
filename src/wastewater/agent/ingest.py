"""Fetch a source, use the LLM tasks to help normalise it, and write the
Parquet + manifest artifacts that ``dashboard.agent_data`` already expects."""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd
from owid.catalog import fetch as _default_catalog_fetch

from .llm_tasks import flag_anomalies, infer_column_mapping, summarize_manifest
from ..search_terms import infer_date_column, read_search_term_file, second_source_column
from .sources import SourceSpec

USER_AGENT = "wastewater-pathogen-data/0.1 (+https://github.com/MalachyReynolds/wastewater-pathogen-data)"
FETCH_TIMEOUT_SECONDS = 120

NORMALIZED_DIR = Path("data") / "normalized"
MANIFESTS_DIR = Path("data_registry") / "manifests"
LATEST_DIR = Path("data_registry") / "latest"


def fetch_source_bytes(source: SourceSpec) -> bytes:
    """Download a source's raw content, matching the other download scripts' conventions."""
    request = Request(source.url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
        return response.read()


def fetch_catalog_frame(source: SourceSpec, fetcher: Any = _default_catalog_fetch) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch a source from the OWID catalog and derive its column mapping from the table's
    own index -- deterministic, so no LLM call is needed here (unlike the free-form CSV path,
    where the schema isn't known in advance).

    ``fetcher`` defaults to ``owid.catalog.fetch`` and is injectable purely for tests, which
    pass a stub returning a small MultiIndex-ed DataFrame instead of a real ``ChartTable``.
    """
    table = fetcher(source.catalog_slug)
    flat = table.reset_index()

    if "dates" in flat.columns:
        date_column = "dates"
    elif "years" in flat.columns:
        date_column = "years"
    else:
        raise ValueError(f"Catalog table '{source.catalog_slug}' has no 'dates' or 'years' index level.")

    geography_column = "entities" if "entities" in flat.columns else None
    signal_columns = [column for column in flat.columns if column not in {date_column, geography_column}]

    return flat, {"date_column": date_column, "geography_column": geography_column, "signal_columns": signal_columns}


def _default_trends_fetch(term: str, geo: str, timeframe: str) -> pd.DataFrame:
    """Fetch a Google Trends interest-over-time series via pytrends.

    pytrends is an unofficial, reverse-engineered client (no official Google Trends search/
    fetch API exists) and is rate-limited by Google in practice -- callers should expect
    occasional failures on rapid repeated calls.
    """
    from pytrends.request import TrendReq

    pytrends = TrendReq(hl="en-GB", tz=0)
    pytrends.build_payload([term], timeframe=timeframe, geo=geo)
    return pytrends.interest_over_time()


def fetch_google_trends_frame(source: SourceSpec, fetcher: Any = _default_trends_fetch) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch a live Google Trends source and derive its column mapping deterministically,
    the same way ``fetch_catalog_frame`` does for OWID -- the shape pytrends returns
    (a ``date`` index plus one value column and ``isPartial``) is known in advance.
    """
    flat = fetcher(source.google_trends_term, source.google_trends_geo, source.google_trends_timeframe).reset_index()
    signal_columns = [column for column in flat.columns if column not in {"date", "isPartial"}]
    return flat, {"date_column": "date", "geography_column": None, "signal_columns": signal_columns}


def fetch_google_trends_local_frame(source: SourceSpec, root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read an already-exported Google Trends CSV from ``Google_trends_v2/`` and derive its
    column mapping using the same conventions as the existing notebook pipeline
    (``search_terms.py``): the date column is inferred, and the predictor is always the
    second column, whatever it's named.
    """
    path = Path(root) / source.google_trends_local_file
    frame = read_search_term_file(path)
    date_column = infer_date_column(frame)
    value_column = second_source_column(frame, source.google_trends_local_file)

    frame = frame.copy()
    frame[value_column] = pd.to_numeric(frame[value_column].astype(str).str.replace(",", "", regex=False), errors="coerce")
    return frame, {"date_column": date_column, "geography_column": None, "signal_columns": [value_column]}


def _year_to_timestamp(year: Any) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(year=int(year), month=1, day=1)
    except (ValueError, OverflowError, TypeError):  # OutOfBoundsDatetime subclasses ValueError
        return None


def _parse_date_column(frame: pd.DataFrame, date_column: str) -> pd.Series:
    """Parse a mapping's date column, handling bare integer years specially.

    ``pd.to_datetime`` on a bare int is misinterpreted as a nanosecond Unix timestamp, not a
    year, and OWID catalog data can include years far outside what ``pd.Timestamp`` can
    represent (e.g. 10,000 BCE) -- so this tolerates per-row failures instead of raising.
    """
    if date_column == "years":
        return frame[date_column].apply(_year_to_timestamp)
    return pd.to_datetime(frame[date_column], errors="coerce")


def _build_normalized_frame(frame: pd.DataFrame, source: SourceSpec, mapping: dict[str, Any]) -> pd.DataFrame:
    date_column = mapping["date_column"]
    geography_column = mapping["geography_column"]
    signal_columns = mapping["signal_columns"]
    if not date_column or date_column not in frame.columns:
        raise ValueError(f"No usable date column found for source '{source.name}'.")

    dates = _parse_date_column(frame, date_column)
    geography = frame[geography_column].astype(str) if geography_column and geography_column in frame.columns else "unknown"

    parts: list[pd.DataFrame] = []
    for column in signal_columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        part = pd.DataFrame(
            {
                "date": dates,
                "value": values,
                "source": source.name,
                "pathogen": source.pathogen,
                "role": source.role,
                "signal_type": column,
                "metric": column,
                "geography_name": geography,
                "geography_code": "unknown",
            }
        ).dropna(subset=["date", "value"])
        if not part.empty:
            parts.append(part)

    if not parts:
        raise ValueError(f"No usable numeric signal columns found for source '{source.name}'.")
    return pd.concat(parts, ignore_index=True).sort_values(["signal_type", "date"]).reset_index(drop=True)


def _compute_stats(normalized: pd.DataFrame, raw_frame: pd.DataFrame, mapping: dict[str, Any]) -> dict[str, Any]:
    signal_columns = [column for column in mapping["signal_columns"] if column in raw_frame.columns]
    missing_fraction = 0.0
    if signal_columns:
        missing_fraction = float(raw_frame[signal_columns].isna().mean().mean())
    return {
        "rows": int(len(normalized)),
        "signal_count": normalized["signal_type"].nunique() if not normalized.empty else 0,
        "date_min": normalized["date"].min().date().isoformat() if not normalized.empty else None,
        "date_max": normalized["date"].max().date().isoformat() if not normalized.empty else None,
        "missing_value_fraction": missing_fraction,
    }


def run_source_ingestion(
    source: SourceSpec,
    root: Path,
    client: Any,
    model: str,
    run_id: str,
    raw_frame: pd.DataFrame | None = None,
    fetcher: Any = _default_catalog_fetch,
    trends_fetcher: Any = _default_trends_fetch,
) -> dict[str, Any]:
    """Ingest one source end to end and return the manifest that was written.

    ``raw_frame`` can be supplied directly (used by tests and ``--local-file``,
    bypassing the network fetch); otherwise it is fetched from ``source.url``,
    ``source.catalog_slug``, ``source.google_trends_term``, or
    ``source.google_trends_local_file``, whichever is set. The catalog and Google
    Trends paths already know their own column mapping (deterministic, from the
    known table shape), so the LLM-based ``infer_column_mapping`` step only runs
    for the URL/CSV path or when a ``raw_frame`` was supplied directly, since that
    frame's shape isn't guaranteed to match any of those conventions.
    """
    root = Path(root)
    mapping: dict[str, Any] | None = None
    if raw_frame is None:
        if source.catalog_slug:
            raw_frame, mapping = fetch_catalog_frame(source, fetcher=fetcher)
        elif source.google_trends_term:
            raw_frame, mapping = fetch_google_trends_frame(source, fetcher=trends_fetcher)
        elif source.google_trends_local_file:
            raw_frame, mapping = fetch_google_trends_local_frame(source, root)
        else:
            raw_frame = pd.read_csv(io.BytesIO(fetch_source_bytes(source)))

    if mapping is None:
        columns = raw_frame.columns.tolist()
        sample_rows = raw_frame.head(5).where(pd.notna(raw_frame.head(5)), None).to_dict(orient="records")
        mapping = infer_column_mapping(client, model, columns, sample_rows)

    normalized = _build_normalized_frame(raw_frame, source, mapping)
    stats = _compute_stats(normalized, raw_frame, mapping)
    validation = flag_anomalies(client, model, stats)
    summary = summarize_manifest(client, model, source.name, stats)

    parquet_path = root / NORMALIZED_DIR / source.name / f"{run_id}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(parquet_path, index=False)

    manifest = {
        "feature_set": source.name,
        "artifact_type": "normalized_signal_table",
        "run_id": run_id,
        "path": parquet_path.relative_to(root).as_posix(),
        "rows": stats["rows"],
        "columns": len(normalized.columns),
        "date_min": stats["date_min"],
        "date_max": stats["date_max"],
        "validation_status": validation["validation_status"],
        "validation_notes": validation["notes"],
        "summary": summary,
        "column_mapping": mapping,
        "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    manifest_path = root / MANIFESTS_DIR / source.name / f"{run_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    latest_path = root / LATEST_DIR / f"{source.name}.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    pointer = {
        "feature_set": source.name,
        "latest_run_id": run_id,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
    }
    latest_path.write_text(json.dumps(pointer, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return manifest

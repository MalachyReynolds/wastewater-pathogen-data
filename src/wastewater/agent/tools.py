"""Tools the chat agent can call.

The two read-only tools (``search_catalog``, ``get_dashboard_status``) execute
immediately and return real results. The two mutating tools
(``propose_add_source``, ``propose_run_ingestion``) only validate their
arguments and describe what *would* happen -- they have no side effects.
Whether that action actually happens is decided by the user confirming it in
the dashboard, not by the model calling the tool. See ``chat.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from owid.catalog import search as _default_searcher

from ..search_terms import (
    DEFAULT_SEARCH_TERMS_PATTERN,
    find_search_term_files,
    read_search_term_file,
    second_source_column,
)
from .sources import SourceSpec

GOOGLE_TRENDS_LOCAL_DIRS = (("1y", "Google_trends_v2/1y_data"), ("5y", "Google_trends_v2/5y_data"))


def _default_trends_suggester(query: str) -> list[dict[str, str]]:
    from pytrends.request import TrendReq

    return TrendReq(hl="en-GB", tz=0).suggestions(query)


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": "Search the OWID (Our World in Data) data catalog for a topic and return candidate datasets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for, e.g. 'influenza hospitalizations'."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_google_trends",
            "description": (
                "Search Google Trends for a topic and return candidate search terms, for fetching a live "
                "interest-over-time series. Note: this is an unofficial, rate-limited API -- repeated calls "
                "in quick succession can fail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for, e.g. 'flu symptoms'."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_local_google_trends_files",
            "description": (
                "Search the Google Trends CSV exports already saved in this repository's Google_trends_v2/ "
                "folder. No network call -- these are one-year and five-year exports someone already downloaded."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term to match, e.g. 'cough'."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dashboard_status",
            "description": "Get a summary of what data and results are currently loaded in the dashboard.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_add_source",
            "description": (
                "Propose adding a new data source for the agent to ingest. This only validates and "
                "describes the proposal -- it does not add anything until the user confirms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "A short, unique, snake_case identifier for this source."},
                    "pathogen": {
                        "type": "string",
                        "description": "The pathogen this source relates to, e.g. 'influenza', 'COVID-19', 'RSV'.",
                    },
                    "role": {
                        "type": "string",
                        "enum": ["predictive", "predicted"],
                        "description": (
                            "'predictive': a leading-indicator input signal (e.g. wastewater levels, search trends). "
                            "'predicted': the outcome being forecast (e.g. hospital admissions, case counts)."
                        ),
                    },
                    "description": {"type": "string", "description": "What this source is and where it comes from."},
                    "url": {"type": "string", "description": "A direct CSV URL, if not using any other location."},
                    "catalog_slug": {
                        "type": "string",
                        "description": "The catalog_slug value from a search_catalog result, if not using any other location.",
                    },
                    "google_trends_term": {
                        "type": "string",
                        "description": (
                            "The 'term' value from a search_google_trends result, if fetching a live Google "
                            "Trends series. Not used with url/catalog_slug/google_trends_local_file."
                        ),
                    },
                    "google_trends_geo": {
                        "type": "string",
                        "description": "ISO country code for the Google Trends fetch, e.g. 'GB'. Empty string means worldwide. Defaults to 'GB'.",
                    },
                    "google_trends_timeframe": {
                        "type": "string",
                        "description": "pytrends timeframe string for the Google Trends fetch, e.g. 'today 12-m' or 'today 5-y'. Defaults to 'today 5-y'.",
                    },
                    "google_trends_local_file": {
                        "type": "string",
                        "description": (
                            "The 'local_file' value from a search_local_google_trends_files result, if using an "
                            "already-downloaded local export instead of a live fetch."
                        ),
                    },
                },
                "required": ["name", "pathogen", "role", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_run_ingestion",
            "description": (
                "Propose running ingestion for an already-known source. This only validates and describes "
                "the proposal -- it does not run anything until the user confirms."
            ),
            "parameters": {
                "type": "object",
                "properties": {"source_name": {"type": "string", "description": "The name of an existing source to ingest."}},
                "required": ["source_name"],
            },
        },
    },
]


def search_google_trends(query: str, searcher: Any = _default_trends_suggester) -> list[dict[str, str]]:
    """Search Google Trends for disambiguated topic candidates via pytrends' suggestions().

    Returns each candidate's ``mid`` as ``term`` -- the value ``propose_add_source`` expects
    for ``google_trends_term`` -- rather than the free-text title, since a bare keyword can be
    ambiguous (e.g. "flu" the illness vs. an unrelated topic) while a topic mid is not.
    """
    results = searcher(query)
    return [{"title": item["title"], "term": item["mid"], "type": item["type"]} for item in results[:5]]


def search_local_google_trends_files(query: str, root: Path, finder: Any = find_search_term_files) -> list[dict[str, str]]:
    """Search the Google Trends CSV exports already saved under Google_trends_v2/.

    The real search term is the file's second column header, not its filename (these exports
    are named by download timestamp, e.g. ``time_series_GB_20250630-1431_...csv``) -- see
    ``search_terms.py``. The repository has duplicate exports of the same term (``(1)``,
    ``(2)`` suffixes); this dedupes to one candidate per (term, period), no network call.
    """
    query_lower = query.lower()
    seen: set[tuple[str, str]] = set()
    matches: list[dict[str, str]] = []
    for period, search_dir in GOOGLE_TRENDS_LOCAL_DIRS:
        files = finder(root, search_dir=search_dir, pattern=DEFAULT_SEARCH_TERMS_PATTERN)
        for rel_path in files["path"]:
            try:
                frame = read_search_term_file(Path(root) / rel_path)
                term = second_source_column(frame, rel_path)
            except Exception:
                continue
            if query_lower not in term.lower():
                continue
            key = (period, term.lower())
            if key in seen:
                continue
            seen.add(key)
            matches.append({"term": term, "local_file": rel_path, "period": period})
    return matches[:10]


def search_catalog(query: str, searcher: Any = _default_searcher) -> list[dict[str, str]]:
    """Search the OWID data catalog and return up to 5 lightweight results.

    Returns each result's full ``.url`` as ``catalog_slug`` rather than its bare ``.slug``.
    A bare slug is ambiguous for "explorer view" results, which can have several distinct
    views sharing one slug (disambiguated by query parameters) -- ``owid.catalog.fetch()``
    only resolves unambiguously when given the full URL. It also matches the field name
    ``propose_add_source`` expects, so the model can pass a result straight through.
    """
    results = searcher(query)
    return [{"title": item.title, "catalog_slug": item.url, "type": item.type} for item in results[:5]]


def get_dashboard_status(context: dict[str, Any]) -> dict[str, Any]:
    """Summarise what's currently loaded in the dashboard, from a pre-built context dict."""
    return {
        "series_panel_loaded": context.get("series_panel_loaded", False),
        "observation_count": context.get("observation_count", 0),
        "series_count": context.get("series_count", 0),
        "latest_manifests": context.get("latest_manifests", []),
    }


def propose_add_source(
    name: str,
    pathogen: str,
    description: str,
    role: str = "",
    url: str | None = None,
    catalog_slug: str | None = None,
    google_trends_term: str | None = None,
    google_trends_geo: str = "GB",
    google_trends_timeframe: str = "today 5-y",
    google_trends_local_file: str | None = None,
) -> dict[str, Any]:
    """Validate a proposed new source. Returns a proposal descriptor, or an error -- never raises."""
    try:
        source = SourceSpec(
            name=name,
            pathogen=pathogen,
            description=description,
            role=role,
            url=url,
            catalog_slug=catalog_slug,
            google_trends_term=google_trends_term,
            google_trends_geo=google_trends_geo,
            google_trends_timeframe=google_trends_timeframe,
            google_trends_local_file=google_trends_local_file,
        )
    except ValueError as exc:
        return {"error": str(exc)}
    return {
        "name": source.name,
        "pathogen": source.pathogen,
        "role": source.role,
        "description": source.description,
        "url": source.url,
        "catalog_slug": source.catalog_slug,
        "google_trends_term": source.google_trends_term,
        "google_trends_geo": source.google_trends_geo,
        "google_trends_timeframe": source.google_trends_timeframe,
        "google_trends_local_file": source.google_trends_local_file,
    }


def propose_run_ingestion(source_name: str, known_source_names: list[str]) -> dict[str, Any]:
    """Validate a proposed ingestion run. Returns a proposal descriptor, or an error -- never raises."""
    if source_name not in known_source_names:
        known = ", ".join(known_source_names) or "(none yet)"
        return {"error": f"Unknown source '{source_name}'. Known sources: {known}"}
    return {"source_name": source_name}

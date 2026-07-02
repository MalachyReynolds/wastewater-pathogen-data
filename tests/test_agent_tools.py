from __future__ import annotations

from pathlib import Path

from wastewater.agent.tools import (
    get_dashboard_status,
    propose_add_source,
    propose_run_ingestion,
    search_catalog,
    search_google_trends,
    search_local_google_trends_files,
)


class _FakeSearchResult:
    def __init__(self, title, url, type_):
        self.title = title
        self.url = url
        self.type = type_


def _fake_searcher(query: str):
    return [
        _FakeSearchResult(
            "Weekly hospital admissions for COVID-19",
            "https://ourworldindata.org/grapher/weekly-hospital-admissions-covid-per-million",
            "chart",
        ),
        _FakeSearchResult(
            "Weekly confirmed cases of influenza",
            "https://ourworldindata.org/explorers/influenza?Interval=Weekly",
            "explorerView",
        ),
    ]


def test_search_catalog_returns_lightweight_results():
    results = search_catalog("covid hospital admissions", searcher=_fake_searcher)

    assert results == [
        {
            "title": "Weekly hospital admissions for COVID-19",
            "catalog_slug": "https://ourworldindata.org/grapher/weekly-hospital-admissions-covid-per-million",
            "type": "chart",
        },
        {
            "title": "Weekly confirmed cases of influenza",
            "catalog_slug": "https://ourworldindata.org/explorers/influenza?Interval=Weekly",
            "type": "explorerView",
        },
    ]


def _fake_trends_suggester(query: str):
    return [
        {"mid": "/m/0cycc", "title": "Flu", "type": "Disease"},
        {"mid": "/m/05_5py4", "title": "Influenza-like illness", "type": "Illness"},
    ]


def test_search_google_trends_returns_mid_as_term():
    results = search_google_trends("influenza", searcher=_fake_trends_suggester)

    assert results == [
        {"title": "Flu", "term": "/m/0cycc", "type": "Disease"},
        {"title": "Influenza-like illness", "term": "/m/05_5py4", "type": "Illness"},
    ]


def _write_trends_csv(root: Path, rel_path: str, term_column: str, rows: list[tuple[str, int]]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'"Time","{term_column}"'] + [f'"{date}",{value}' for date, value in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_search_local_google_trends_files_matches_second_column_term(tmp_path: Path):
    _write_trends_csv(tmp_path, "Google_trends_v2/1y_data/time_series_GB_a.csv", "persistent cough", [("2026-01-04", 37)])
    _write_trends_csv(tmp_path, "Google_trends_v2/5y_data/time_series_GB_b.csv", "persistent cough", [("2021-07-01", 41)])
    _write_trends_csv(tmp_path, "Google_trends_v2/1y_data/time_series_GB_c.csv", "sore throat", [("2026-01-04", 12)])

    results = search_local_google_trends_files("cough", tmp_path)

    # read_search_term_file normalises column headers (lowercase, spaces -> underscores),
    # matching how fetch_google_trends_local_frame reads the same files at ingest time.
    assert results == [
        {"term": "persistent_cough", "local_file": "Google_trends_v2/1y_data/time_series_GB_a.csv", "period": "1y"},
        {"term": "persistent_cough", "local_file": "Google_trends_v2/5y_data/time_series_GB_b.csv", "period": "5y"},
    ]


def test_search_local_google_trends_files_dedupes_duplicate_exports(tmp_path: Path):
    _write_trends_csv(tmp_path, "Google_trends_v2/1y_data/time_series_GB_a.csv", "cough", [("2026-01-04", 37)])
    _write_trends_csv(tmp_path, "Google_trends_v2/1y_data/time_series_GB_a (1).csv", "cough", [("2026-01-04", 37)])

    results = search_local_google_trends_files("cough", tmp_path)

    assert len(results) == 1


def test_get_dashboard_status_applies_defaults_for_missing_context_keys():
    status = get_dashboard_status({})

    assert status == {
        "series_panel_loaded": False,
        "observation_count": 0,
        "series_count": 0,
        "latest_manifests": [],
    }


def test_get_dashboard_status_passes_through_provided_values():
    context = {"series_panel_loaded": True, "observation_count": 100, "series_count": 5, "latest_manifests": [{"a": 1}]}
    assert get_dashboard_status(context) == context


def test_propose_add_source_valid_url():
    proposal = propose_add_source(name="x", pathogen="RSV", role="predictive", description="test", url="https://example.invalid/data.csv")
    assert proposal == {
        "name": "x",
        "pathogen": "RSV",
        "role": "predictive",
        "description": "test",
        "url": "https://example.invalid/data.csv",
        "catalog_slug": None,
        "google_trends_term": None,
        "google_trends_geo": "GB",
        "google_trends_timeframe": "today 5-y",
        "google_trends_local_file": None,
    }


def test_propose_add_source_valid_catalog_slug():
    proposal = propose_add_source(name="x", pathogen="RSV", role="predicted", description="test", catalog_slug="some-slug")
    assert proposal["catalog_slug"] == "some-slug"
    assert proposal["url"] is None
    assert proposal["role"] == "predicted"


def test_propose_add_source_valid_google_trends_term():
    proposal = propose_add_source(
        name="x", pathogen="influenza", role="predictive", description="test", google_trends_term="/m/0cycc", google_trends_geo="US"
    )
    assert proposal["google_trends_term"] == "/m/0cycc"
    assert proposal["google_trends_geo"] == "US"
    assert proposal["google_trends_timeframe"] == "today 5-y"
    assert proposal["url"] is None
    assert proposal["catalog_slug"] is None


def test_propose_add_source_valid_google_trends_local_file():
    proposal = propose_add_source(
        name="x",
        pathogen="respiratory",
        role="predictive",
        description="test",
        google_trends_local_file="Google_trends_v2/1y_data/time_series_GB_x.csv",
    )
    assert proposal["google_trends_local_file"] == "Google_trends_v2/1y_data/time_series_GB_x.csv"
    assert proposal["google_trends_term"] is None


def test_propose_add_source_rejects_neither_url_nor_slug():
    proposal = propose_add_source(name="x", pathogen="RSV", role="predictive", description="test")
    assert "error" in proposal


def test_propose_add_source_rejects_both_url_and_slug():
    proposal = propose_add_source(
        name="x", pathogen="RSV", role="predictive", description="test", url="https://example.invalid/data.csv", catalog_slug="some-slug"
    )
    assert "error" in proposal


def test_propose_add_source_rejects_invalid_role():
    proposal = propose_add_source(name="x", pathogen="RSV", role="unsure", description="test", url="https://example.invalid/data.csv")
    assert "error" in proposal


def test_propose_run_ingestion_known_source():
    proposal = propose_run_ingestion("existing_source", known_source_names=["existing_source", "other"])
    assert proposal == {"source_name": "existing_source"}


def test_propose_run_ingestion_unknown_source():
    proposal = propose_run_ingestion("missing_source", known_source_names=["existing_source"])
    assert "error" in proposal

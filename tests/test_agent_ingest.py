from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wastewater.agent.ingest import (
    _year_to_timestamp,
    fetch_catalog_frame,
    fetch_google_trends_frame,
    fetch_google_trends_local_frame,
    run_source_ingestion,
)
from wastewater.agent.sources import SourceSpec
from wastewater.dashboard.agent_data import (
    list_latest_agent_manifests,
    load_normalized_signal_tables,
    normalized_signals_to_canonical_series,
)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletionResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def create(self, **kwargs):
        # No LLM available in tests -- return something unparseable so every
        # llm_tasks function falls back to its heuristic, matching how the
        # pipeline behaves if Nebius is unreachable.
        return _FakeCompletionResponse("not valid json")


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = _FakeChat()


def _synthetic_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-01-04", "2026-01-11", "2026-01-18", "2026-01-25"],
            "location": ["Testland", "Testland", "Testland", "Testland"],
            "new_cases": [10.0, 12.0, 15.0, 9.0],
            "new_deaths": [1.0, 0.0, 2.0, 1.0],
        }
    )


def test_run_source_ingestion_round_trips_through_agent_data_readers(tmp_path: Path):
    source = SourceSpec(
        name="test_source",
        url="https://example.invalid/data.csv",
        pathogen="COVID-19",
        role="predicted",
        description="test",
    )

    manifest = run_source_ingestion(
        source,
        root=tmp_path,
        client=FakeClient(),
        model="test-model",
        run_id="20260702T120000Z",
        raw_frame=_synthetic_raw_frame(),
    )

    assert manifest["feature_set"] == "test_source"
    assert manifest["run_id"] == "20260702T120000Z"
    assert manifest["rows"] == 8  # 4 dates x 2 signal columns
    assert manifest["validation_status"] in {"passed", "warning", "failed"}
    assert manifest["summary"]

    manifests = list_latest_agent_manifests(tmp_path)
    assert manifests.loc[0, "feature_set"] == "test_source"
    assert manifests.loc[0, "artifact_type"] == "normalized_signal_table"

    normalized = load_normalized_signal_tables(tmp_path)
    assert not normalized.empty
    assert set(normalized["signal_type"]) == {"new_cases", "new_deaths"}
    assert set(normalized["role"]) == {"predicted"}

    # Neither "new_cases" nor "new_deaths" contains "admission" -- the old heuristic would have
    # guessed "predictive" here. Proving the series come back "predicted" confirms the explicit
    # classification is what's actually used, not the substring guess.
    series = normalized_signals_to_canonical_series(normalized)
    assert not series.empty
    assert set(series["role"]) == {"predicted"}
    assert series["series_id"].nunique() == 2


def test_source_spec_requires_exactly_one_location_field():
    with pytest.raises(ValueError):
        SourceSpec(name="x", pathogen="COVID-19", role="predictive", description="test")

    with pytest.raises(ValueError):
        SourceSpec(
            name="x",
            pathogen="COVID-19",
            role="predictive",
            description="test",
            url="https://example.invalid/data.csv",
            catalog_slug="some-slug",
        )

    with pytest.raises(ValueError):
        SourceSpec(
            name="x",
            pathogen="COVID-19",
            role="predictive",
            description="test",
            url="https://example.invalid/data.csv",
            google_trends_term="/m/0cycc",
        )

    # exactly one set should not raise
    SourceSpec(name="x", pathogen="COVID-19", role="predictive", description="test", url="https://example.invalid/data.csv")
    SourceSpec(name="x", pathogen="COVID-19", role="predictive", description="test", catalog_slug="some-slug")
    SourceSpec(name="x", pathogen="COVID-19", role="predictive", description="test", google_trends_term="/m/0cycc")
    SourceSpec(
        name="x", pathogen="COVID-19", role="predictive", description="test", google_trends_local_file="Google_trends_v2/1y_data/x.csv"
    )


def test_source_spec_requires_a_valid_role():
    with pytest.raises(ValueError):
        SourceSpec(name="x", pathogen="COVID-19", role="unsure", description="test", url="https://example.invalid/data.csv")


def _fake_daily_catalog_fetcher(slug: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"weekly_admissions_hosp_per_1m": [2.917, 8.150, 12.955]},
        index=pd.MultiIndex.from_tuples(
            [("Belgium", "2020-03-12"), ("Belgium", "2020-03-13"), ("Belgium", "2020-03-14")],
            names=["entities", "dates"],
        ),
    )


def _fake_annual_catalog_fetcher(slug: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"population_historical": [14737, 20405, 28253]},
        index=pd.MultiIndex.from_tuples(
            [("Afghanistan", -10000), ("Afghanistan", -9000), ("Afghanistan", 2020)],
            names=["entities", "years"],
        ),
    )


def test_fetch_catalog_frame_maps_daily_dates():
    source = SourceSpec(
        name="x",
        pathogen="COVID-19",
        role="predicted",
        description="test",
        catalog_slug="weekly-hospital-admissions-covid-per-million",
    )
    flat, mapping = fetch_catalog_frame(source, fetcher=_fake_daily_catalog_fetcher)

    assert mapping == {
        "date_column": "dates",
        "geography_column": "entities",
        "signal_columns": ["weekly_admissions_hosp_per_1m"],
    }
    assert len(flat) == 3


def test_fetch_catalog_frame_maps_annual_years():
    source = SourceSpec(name="x", pathogen="n/a", role="predictive", description="test", catalog_slug="population")
    flat, mapping = fetch_catalog_frame(source, fetcher=_fake_annual_catalog_fetcher)

    assert mapping == {
        "date_column": "years",
        "geography_column": "entities",
        "signal_columns": ["population_historical"],
    }
    assert len(flat) == 3


def test_year_to_timestamp_handles_normal_and_edge_case_years():
    assert _year_to_timestamp(2020) == pd.Timestamp(year=2020, month=1, day=1)
    assert _year_to_timestamp(-10000) is None  # BCE, unrepresentable
    assert _year_to_timestamp(999999999) is None  # far future, out of pandas' range


def test_run_source_ingestion_round_trips_catalog_source_through_agent_data_readers(tmp_path: Path):
    source = SourceSpec(
        name="catalog_source",
        pathogen="COVID-19",
        role="predicted",
        description="test",
        catalog_slug="weekly-hospital-admissions-covid-per-million",
    )

    manifest = run_source_ingestion(
        source,
        root=tmp_path,
        client=FakeClient(),
        model="test-model",
        run_id="20260702T120000Z",
        fetcher=_fake_daily_catalog_fetcher,
    )

    assert manifest["feature_set"] == "catalog_source"
    assert manifest["column_mapping"]["date_column"] == "dates"
    assert manifest["rows"] == 3

    manifests = list_latest_agent_manifests(tmp_path)
    assert manifests.loc[0, "feature_set"] == "catalog_source"

    normalized = load_normalized_signal_tables(tmp_path)
    assert not normalized.empty
    assert set(normalized["signal_type"]) == {"weekly_admissions_hosp_per_1m"}
    assert set(normalized["geography_name"]) == {"Belgium"}
    assert set(normalized["role"]) == {"predicted"}

    series = normalized_signals_to_canonical_series(normalized)
    assert not series.empty
    assert set(series["role"]) == {"predicted"}


def _fake_trends_fetcher(term: str, geo: str, timeframe: str) -> pd.DataFrame:
    assert term == "/m/0cycc"
    assert geo == "GB"
    assert timeframe == "today 5-y"
    return pd.DataFrame(
        {"/m/0cycc": [11, 8, 10], "isPartial": [False, False, True]},
        index=pd.DatetimeIndex(["2026-01-04", "2026-01-11", "2026-01-18"], name="date"),
    )


def test_fetch_google_trends_frame_maps_date_and_drops_ispartial():
    source = SourceSpec(name="x", pathogen="influenza", role="predictive", description="test", google_trends_term="/m/0cycc")
    flat, mapping = fetch_google_trends_frame(source, fetcher=_fake_trends_fetcher)

    assert mapping == {"date_column": "date", "geography_column": None, "signal_columns": ["/m/0cycc"]}
    assert len(flat) == 3
    assert "isPartial" not in mapping["signal_columns"]


def test_run_source_ingestion_round_trips_google_trends_source(tmp_path: Path):
    source = SourceSpec(name="trends_source", pathogen="influenza", role="predictive", description="test", google_trends_term="/m/0cycc")

    manifest = run_source_ingestion(
        source,
        root=tmp_path,
        client=FakeClient(),
        model="test-model",
        run_id="20260702T120000Z",
        trends_fetcher=_fake_trends_fetcher,
    )

    assert manifest["feature_set"] == "trends_source"
    assert manifest["rows"] == 3

    normalized = load_normalized_signal_tables(tmp_path)
    assert not normalized.empty
    assert set(normalized["role"]) == {"predictive"}


def test_fetch_google_trends_local_frame_reads_second_column_as_signal(tmp_path: Path):
    search_dir = tmp_path / "Google_trends_v2" / "1y_data"
    search_dir.mkdir(parents=True)
    rel_path = "Google_trends_v2/1y_data/time_series_GB_test.csv"
    (tmp_path / rel_path).write_text('"Time","persistent cough"\n"2026-01-04",37\n"2026-01-11",27\n', encoding="utf-8")

    source = SourceSpec(
        name="x", pathogen="respiratory", role="predictive", description="test", google_trends_local_file=rel_path
    )
    flat, mapping = fetch_google_trends_local_frame(source, tmp_path)

    assert mapping == {"date_column": "time", "geography_column": None, "signal_columns": ["persistent_cough"]}
    assert list(flat["persistent_cough"]) == [37.0, 27.0]


def test_run_source_ingestion_round_trips_local_google_trends_source(tmp_path: Path):
    search_dir = tmp_path / "Google_trends_v2" / "1y_data"
    search_dir.mkdir(parents=True)
    rel_path = "Google_trends_v2/1y_data/time_series_GB_test.csv"
    (tmp_path / rel_path).write_text('"Time","persistent cough"\n"2026-01-04",37\n"2026-01-11",27\n', encoding="utf-8")

    source = SourceSpec(
        name="local_trends_source", pathogen="respiratory", role="predictive", description="test", google_trends_local_file=rel_path
    )

    manifest = run_source_ingestion(
        source, root=tmp_path, client=FakeClient(), model="test-model", run_id="20260702T120000Z"
    )

    assert manifest["feature_set"] == "local_trends_source"
    assert manifest["rows"] == 2

    normalized = load_normalized_signal_tables(tmp_path)
    assert not normalized.empty
    assert set(normalized["role"]) == {"predictive"}

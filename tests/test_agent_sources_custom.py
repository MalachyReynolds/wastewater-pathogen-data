from __future__ import annotations

import json
from pathlib import Path

import pytest

from wastewater.agent.sources import SourceSpec, add_custom_source, list_sources, load_custom_sources


def test_load_custom_sources_returns_empty_list_when_no_file(tmp_path: Path):
    assert load_custom_sources(tmp_path) == []


def test_add_custom_source_persists_and_round_trips(tmp_path: Path):
    source = SourceSpec(name="custom_a", pathogen="RSV", role="predictive", description="test", url="https://example.invalid/data.csv")

    add_custom_source(source, tmp_path)

    loaded = load_custom_sources(tmp_path)
    assert loaded == [source]

    assert source in list_sources(tmp_path)


def test_add_custom_source_rejects_duplicate_name(tmp_path: Path):
    source = SourceSpec(name="custom_a", pathogen="RSV", role="predictive", description="test", url="https://example.invalid/data.csv")
    add_custom_source(source, tmp_path)

    duplicate = SourceSpec(name="custom_a", pathogen="influenza", role="predicted", description="different", catalog_slug="some-slug")
    with pytest.raises(ValueError):
        add_custom_source(duplicate, tmp_path)


def test_list_sources_combines_multiple_custom_sources(tmp_path: Path):
    first = SourceSpec(name="a", pathogen="RSV", role="predictive", description="test", url="https://example.invalid/a.csv")
    second = SourceSpec(name="b", pathogen="influenza", role="predicted", description="test", catalog_slug="some-slug")

    add_custom_source(first, tmp_path)
    add_custom_source(second, tmp_path)

    names = {source.name for source in list_sources(tmp_path)}
    assert names == {"a", "b"}


def test_load_custom_sources_defaults_legacy_entries_missing_role(tmp_path: Path):
    registry_dir = tmp_path / "data_registry"
    registry_dir.mkdir()
    (registry_dir / "custom_sources.json").write_text(
        json.dumps([{"name": "legacy", "pathogen": "RSV", "description": "test", "url": "https://example.invalid/data.csv", "catalog_slug": None}])
    )

    loaded = load_custom_sources(tmp_path)

    assert len(loaded) == 1
    assert loaded[0].role == "predictive"

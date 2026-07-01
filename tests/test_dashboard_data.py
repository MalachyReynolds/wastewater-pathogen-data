from __future__ import annotations

import pandas as pd
import pytest

from wastewater.dashboard.data import build_custom_series, list_series_catalogue, merge_series, summarise_dataset


def _canonical_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-01", "2024-01-08"]
            ),
            "value": [1.0, 2.0, 3.0, 10.0, 11.0],
            "series_id": ["a", "a", "a", "b", "b"],
            "role": ["predictive", "predictive", "predictive", "predicted", "predicted"],
            "dataset_family": ["fam_a", "fam_a", "fam_a", "fam_b", "fam_b"],
            "series_name": ["Series A", "Series A", "Series A", "Series B", "Series B"],
            "source_file": ["a.csv", "a.csv", "a.csv", "b.csv", "b.csv"],
        }
    )


def test_list_series_catalogue_summarises_each_series():
    catalogue = list_series_catalogue(_canonical_frame())

    assert set(catalogue["series_id"]) == {"a", "b"}
    row_a = catalogue.loc[catalogue["series_id"] == "a"].iloc[0]
    assert row_a["role"] == "predictive"
    assert row_a["n_obs"] == 3
    assert row_a["date_min"] == pd.Timestamp("2024-01-01")
    assert row_a["date_max"] == pd.Timestamp("2024-01-15")

    row_b = catalogue.loc[catalogue["series_id"] == "b"].iloc[0]
    assert row_b["n_obs"] == 2


def test_list_series_catalogue_handles_empty_frame():
    catalogue = list_series_catalogue(pd.DataFrame())
    assert list(catalogue.columns) == [
        "series_id",
        "role",
        "dataset_family",
        "series_name",
        "n_obs",
        "date_min",
        "date_max",
    ]
    assert catalogue.empty


def test_summarise_dataset_reports_missing_values():
    frame = pd.DataFrame({"x": [1.0, None, 3.0], "y": [1, 2, 3]})
    summary = summarise_dataset(frame)
    assert summary.loc["x", "missing"] == 1
    assert summary.loc["y", "missing"] == 0


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "When": ["2024-02-01", "2024-02-08", "not-a-date", "2024-02-15"],
            "Cases": [10.0, None, 12.0, 14.0],
        }
    )


def test_build_custom_series_produces_canonical_columns_and_drops_invalid_rows():
    custom = build_custom_series(
        _raw_frame(),
        date_column="When",
        value_column="Cases",
        series_id="custom::cases",
        series_name="Cases",
        role="predictive",
        source_file="raw.csv",
    )

    assert list(custom.columns) == ["date", "value", "series_id", "role", "dataset_family", "series_name", "source_file"]
    assert len(custom) == 2
    assert (custom["series_id"] == "custom::cases").all()
    assert (custom["role"] == "predictive").all()


def test_build_custom_series_rejects_invalid_role():
    with pytest.raises(ValueError):
        build_custom_series(
            _raw_frame(), date_column="When", value_column="Cases", series_id="x", series_name="X", role="bogus"
        )


def test_merge_series_appends_new_series_id():
    base = _canonical_frame()
    custom = build_custom_series(
        _raw_frame(), date_column="When", value_column="Cases", series_id="custom::cases", series_name="Cases", role="predictive"
    )
    merged = merge_series(base, custom)

    assert set(merged["series_id"]) == {"a", "b", "custom::cases"}
    assert len(merged) == len(base) + len(custom)


def test_merge_series_replaces_existing_series_id():
    base = _canonical_frame()
    custom_v1 = build_custom_series(
        _raw_frame(), date_column="When", value_column="Cases", series_id="a", series_name="A replacement", role="predictive"
    )
    merged = merge_series(base, custom_v1)

    assert (merged.loc[merged["series_id"] == "a", "series_name"] == "A replacement").all()
    assert (merged["series_id"] == "a").sum() == len(custom_v1)


def test_merge_series_with_no_existing_panel_returns_new_series():
    custom = build_custom_series(
        _raw_frame(), date_column="When", value_column="Cases", series_id="custom::cases", series_name="Cases", role="predicted"
    )
    merged = merge_series(None, custom)
    assert merged.equals(custom.reset_index(drop=True))

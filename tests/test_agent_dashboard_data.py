from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wastewater.dashboard.agent_data import (  # noqa: E402
    feature_table_to_canonical_series,
    list_latest_agent_manifests,
    load_feature_table,
    normalized_signals_to_canonical_series,
)


def test_feature_manifest_loading_and_canonical_conversion(tmp_path: Path) -> None:
    feature_set = "england_rsv_admissions_2w"
    run_id = "20260702T120000Z"

    feature_dir = tmp_path / "data" / "features" / f"feature_set={feature_set}" / f"run_id={run_id}"
    feature_dir.mkdir(parents=True)
    feature_path = feature_dir / "features.parquet"
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4, freq="W"),
            "target_rsv_admissions_t_plus_2": [1.0, 2.0, 3.0, 4.0],
            "rsv_positivity_lag_1w": [0.1, 0.2, 0.3, 0.4],
        }
    ).to_parquet(feature_path)

    manifest_dir = tmp_path / "data_registry" / "manifests" / feature_set
    manifest_dir.mkdir(parents=True)
    manifest_path = manifest_dir / f"{run_id}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "feature_set": feature_set,
                "artifact_type": "feature_table",
                "run_id": run_id,
                "path": feature_path.relative_to(tmp_path).as_posix(),
                "rows": 4,
                "columns": 3,
                "validation_status": "passed",
            }
        )
    )

    latest_dir = tmp_path / "data_registry" / "latest"
    latest_dir.mkdir(parents=True)
    (latest_dir / f"{feature_set}.json").write_text(
        json.dumps(
            {
                "feature_set": feature_set,
                "latest_run_id": run_id,
                "manifest_path": manifest_path.relative_to(tmp_path).as_posix(),
            }
        )
    )

    manifests = list_latest_agent_manifests(tmp_path)
    assert manifests.loc[0, "feature_set"] == feature_set

    frame, manifest = load_feature_table(tmp_path, feature_set)
    assert manifest["run_id"] == run_id
    assert len(frame) == 4

    series = feature_table_to_canonical_series(
        frame,
        feature_set=feature_set,
        source_file=manifest["path"],
    )
    assert set(series["role"]) == {"predicted", "predictive"}
    assert series["series_id"].nunique() == 2


def test_normalized_signals_to_canonical_series() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3, freq="W"),
            "value": [1.0, 2.0, 3.0],
            "source": "ukhsa",
            "pathogen": "RSV",
            "signal_type": "hospital_admissions",
            "metric": "admission_rate",
            "geography_name": "England",
            "geography_code": "E92000001",
        }
    )

    series = normalized_signals_to_canonical_series(frame)
    assert len(series) == 3
    assert series["role"].iloc[0] == "predicted"
    assert series["dataset_family"].iloc[0] == "agent_normalized_signal"


def test_normalized_signals_to_canonical_series_prefers_explicit_role_over_heuristic() -> None:
    # "hospital_admissions"/"admission_rate" would make the substring heuristic guess
    # "predicted" -- an explicit role column set to "predictive" should win regardless.
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3, freq="W"),
            "value": [1.0, 2.0, 3.0],
            "source": "ukhsa",
            "pathogen": "RSV",
            "role": "predictive",
            "signal_type": "hospital_admissions",
            "metric": "admission_rate",
            "geography_name": "England",
            "geography_code": "E92000001",
        }
    )

    series = normalized_signals_to_canonical_series(frame)
    assert len(series) == 3
    assert series["role"].iloc[0] == "predictive"

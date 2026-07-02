from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st

from wastewater.dashboard.data import list_series_catalogue

ROOT = Path(__file__).resolve().parent.parent
PAGE_FILES = sorted((ROOT / "pages").glob("*.py"))


def _seed_canonical_series() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=20, freq="W")
    predictor = pd.DataFrame(
        {
            "date": dates,
            "value": range(20),
            "series_id": "predictor_a",
            "role": "predictive",
            "dataset_family": "fam_p",
            "series_name": "Predictor A",
            "source_file": "p.csv",
        }
    )
    target = pd.DataFrame(
        {
            "date": dates,
            "value": range(20),
            "series_id": "target_a",
            "role": "predicted",
            "dataset_family": "fam_t",
            "series_name": "Target A",
            "source_file": "t.csv",
        }
    )
    return pd.concat([predictor, target], ignore_index=True)


@pytest.mark.parametrize("page_path", PAGE_FILES, ids=[path.name for path in PAGE_FILES])
def test_page_imports(page_path: Path):
    series = _seed_canonical_series()
    st.session_state["series"] = series
    st.session_state["catalogue"] = list_series_catalogue(series)

    spec = importlib.util.spec_from_file_location(f"dashboard_page_{page_path.stem}", page_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

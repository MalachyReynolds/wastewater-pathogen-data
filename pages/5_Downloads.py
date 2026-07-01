from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wastewater.dashboard.downloads import (
    DOWNLOAD_JOBS,
    download_external_sources,
    poll_background_job,
    start_background_download,
)
from wastewater.dashboard.export import dataframe_download_buttons, file_download_button

st.set_page_config(page_title="Downloads", page_icon="\U0001f4e5", layout="wide")
st.title("Downloads")

st.header("Refresh source data")

st.subheader("External sources (OWID + weather)")
st.caption("Fast, no API key required.")
if st.button("Refresh external sources"):
    with st.spinner("Downloading OWID COVID data and Open-Meteo weather..."):
        try:
            result = download_external_sources(ROOT)
            st.success(f"Downloaded {sum(len(v) for v in result.values())} external files.")
        except Exception as exc:
            st.error(f"External download failed: {exc}")

st.subheader("Slower, network-heavy sources")
for job_key, job in DOWNLOAD_JOBS.items():
    st.markdown(f"**{job.label}**")
    st.caption(job.warning)
    session_key = f"download_job_{job_key}"

    if session_key not in st.session_state:
        if st.button("Start", key=f"start_{job_key}"):
            st.session_state[session_key] = start_background_download(job_key, ROOT)
            st.rerun()
    else:
        background_job = poll_background_job(st.session_state[session_key])
        if background_job.status == "running":
            st.status(f"Running ({job.script})...", state="running")
            if st.button("Check status", key=f"check_{job_key}"):
                st.rerun()
        elif background_job.status == "done":
            st.status("Finished", state="complete")
        else:
            st.status(f"Failed (exit code {background_job.return_code})", state="error")

        with st.expander("Log"):
            st.code("\n".join(background_job.log_lines[-100:]))

        if background_job.status != "running":
            for output_file in job.output_files:
                file_download_button(ROOT / output_file, f"Download {output_file}", key=f"dl_{job_key}_{output_file}")
            if st.button("Clear", key=f"clear_{job_key}"):
                del st.session_state[session_key]
                st.rerun()

st.divider()
st.header("Export")

if "series" in st.session_state:
    dataframe_download_buttons(st.session_state["series"], "Canonical series panel", "series_panel", key="export_series")
if "raw_frame" in st.session_state:
    dataframe_download_buttons(st.session_state["raw_frame"], "Raw dataset preview", "raw_dataset", key="export_raw")
if "last_model_predictions" in st.session_state:
    dataframe_download_buttons(
        st.session_state["last_model_predictions"], "Model predictions", "model_predictions", key="export_model_predictions"
    )
if "last_forecast_predictions" in st.session_state:
    dataframe_download_buttons(
        st.session_state["last_forecast_predictions"], "Forecast predictions", "forecast_predictions", key="export_forecast"
    )

if not any(
    key in st.session_state
    for key in ["series", "raw_frame", "last_model_predictions", "last_forecast_predictions"]
):
    st.info("Load data or run a model/forecast on the other pages to enable exports here.")

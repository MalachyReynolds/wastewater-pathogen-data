from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wastewater.agent.chat import execute_confirmed_action, run_chat_turn
from wastewater.agent.chat import SYSTEM_PROMPT as AGENT_SYSTEM_PROMPT
from wastewater.agent.config import load_config
from wastewater.agent.nebius_client import forge_nebius_client
from wastewater.agent.sources import list_sources
from wastewater.dashboard.agent_data import (
    feature_table_to_canonical_series,
    list_latest_agent_manifests,
    load_feature_table,
    load_normalized_signal_tables,
    normalized_signals_to_canonical_series,
)
from wastewater.dashboard.compat import make_streamlit_safe
from wastewater.dashboard.data import list_series_catalogue, merge_series
from wastewater.dashboard.downloads import DOWNLOAD_JOBS, poll_background_job, start_background_download

st.set_page_config(page_title="Agent Data", page_icon="🤖", layout="wide")
st.title("Agent Data")
st.write(
    "Load versioned Parquet outputs produced by the autonomous respiratory data agent. "
    "The loaded artifacts are adapted into the dashboard's canonical series panel, so the "
    "Explore, Model, and Forecast pages can use them directly."
)

st.info(
    "Expected layout: `data_registry/latest/*.json` points to feature-table manifests, "
    "and manifests point to Parquet files under `data/features/`. Normalized signal tables "
    "can also be loaded from `data/normalized/**/*.parquet`."
)


@st.cache_data(show_spinner=False)
def cached_latest_manifests(root: str) -> pd.DataFrame:
    return list_latest_agent_manifests(Path(root))


@st.cache_data(show_spinner=False)
def cached_feature_table(root: str, feature_set: str) -> tuple[pd.DataFrame, dict]:
    return load_feature_table(Path(root), feature_set)


@st.cache_data(show_spinner=False)
def cached_normalized_signals(root: str) -> pd.DataFrame:
    return load_normalized_signal_tables(Path(root))


st.header("Run the data agent")
agent_job = DOWNLOAD_JOBS["data_agent"]
st.caption(agent_job.warning)
agent_session_key = "download_job_data_agent"

if agent_session_key not in st.session_state:
    if st.button("Start data agent run"):
        st.session_state[agent_session_key] = start_background_download("data_agent", ROOT)
        st.rerun()
else:
    agent_job_state = poll_background_job(st.session_state[agent_session_key])
    if agent_job_state.status == "running":
        st.status("Running the data agent...", state="running")
        if st.button("Check status"):
            st.rerun()
    elif agent_job_state.status == "done":
        st.status("Finished", state="complete")
    else:
        st.status(f"Failed (exit code {agent_job_state.return_code})", state="error")

    with st.expander("Log"):
        st.code("\n".join(agent_job_state.log_lines[-100:]))

    if agent_job_state.status != "running" and st.button("Clear"):
        del st.session_state[agent_session_key]
        st.rerun()

st.divider()
st.header("Chat with the agent")
st.caption(
    "Ask it to search for data sources, add one, or run ingestion, or ask what's currently "
    "loaded. Actions that write files or spend API calls always ask for your confirmation first."
)

if "agent_chat_messages" not in st.session_state:
    st.session_state["agent_chat_messages"] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
if "agent_chat_pending_action" not in st.session_state:
    st.session_state["agent_chat_pending_action"] = None

for message in st.session_state["agent_chat_messages"]:
    if message["role"] in ("user", "assistant") and message.get("content"):
        with st.chat_message(message["role"]):
            st.write(message["content"])

pending_action = st.session_state["agent_chat_pending_action"]
if pending_action:
    with st.chat_message("assistant"):
        st.warning(pending_action["summary"])
        role_override = None
        if pending_action["name"] == "propose_add_source":
            proposed_role = pending_action["proposal"].get("role", "predictive")
            role_override = st.radio(
                "Classify this dataset before adding it",
                ["predictive", "predicted"],
                index=0 if proposed_role == "predictive" else 1,
                key="agent_chat_role_override",
                help=(
                    "predictive: a leading-indicator input signal (e.g. wastewater levels, search trends). "
                    "predicted: the outcome being forecast (e.g. hospital admissions, case counts)."
                ),
            )
        confirm_col, cancel_col = st.columns(2)
        if confirm_col.button("Confirm", key="agent_chat_confirm"):
            try:
                config = load_config()
                client = forge_nebius_client(config)
                result_text = execute_confirmed_action(
                    pending_action, ROOT, client, config.model, role_override=role_override
                )
            except Exception as exc:
                result_text = f"Action failed: {exc}"
            st.session_state["agent_chat_messages"].append({"role": "assistant", "content": result_text})
            st.session_state["agent_chat_pending_action"] = None
            cached_latest_manifests.clear()
            cached_normalized_signals.clear()
            st.rerun()
        if cancel_col.button("Cancel", key="agent_chat_cancel"):
            st.session_state["agent_chat_messages"].append({"role": "assistant", "content": "Cancelled."})
            st.session_state["agent_chat_pending_action"] = None
            st.rerun()

chat_input = st.chat_input("Ask the agent...", disabled=bool(pending_action))
if chat_input:
    try:
        config = load_config()
        client = forge_nebius_client(config)
    except RuntimeError as exc:
        st.error(str(exc))
    else:
        loaded_series = st.session_state.get("series")
        loaded_catalogue = st.session_state.get("catalogue")
        context = {
            "series_panel_loaded": loaded_series is not None,
            "observation_count": len(loaded_series) if loaded_series is not None else 0,
            "series_count": len(loaded_catalogue) if loaded_catalogue is not None else 0,
            "latest_manifests": cached_latest_manifests(str(ROOT)).to_dict(orient="records"),
        }
        known_source_names = [source.name for source in list_sources(ROOT)]

        st.session_state["agent_chat_messages"].append({"role": "user", "content": chat_input})
        result = run_chat_turn(
            client,
            config.model,
            st.session_state["agent_chat_messages"],
            context,
            known_source_names,
            root=ROOT,
        )
        st.session_state["agent_chat_messages"] = result.messages
        st.session_state["agent_chat_pending_action"] = result.pending_action
        st.rerun()

with st.expander("Known sources"):
    known_sources = list_sources(ROOT)
    if not known_sources:
        st.caption("No sources known yet -- ask the chat to add one, or add a SourceSpec in code.")
    else:
        st.dataframe(
            make_streamlit_safe(
                pd.DataFrame(
                    [
                        {
                            "name": s.name,
                            "pathogen": s.pathogen,
                            "role": s.role,
                            "location": s.url or s.catalog_slug or s.google_trends_term or s.google_trends_local_file,
                        }
                        for s in known_sources
                    ]
                )
            ),
            width="stretch",
        )

st.divider()
st.header("Latest registered feature tables")
manifest_table = cached_latest_manifests(str(ROOT))

if manifest_table.empty:
    st.warning(
        "No latest agent manifests were found yet. Run the agent storage pipeline first, "
        "or place a latest pointer under `data_registry/latest/`."
    )
else:
    st.dataframe(make_streamlit_safe(manifest_table), width="stretch")
    feature_sets = manifest_table["feature_set"].dropna().astype(str).tolist()
    selected_feature_set = st.selectbox("Feature table", feature_sets)

    if st.button("Load selected feature table"):
        try:
            feature_frame, manifest = cached_feature_table(str(ROOT), selected_feature_set)
            st.session_state["agent_feature_frame"] = feature_frame
            st.session_state["agent_feature_manifest"] = manifest
            st.success(f"Loaded {selected_feature_set}: {len(feature_frame):,} rows, {len(feature_frame.columns):,} columns.")
        except Exception as exc:
            st.error(f"Could not load feature table: {exc}")

if "agent_feature_frame" in st.session_state:
    feature_frame = st.session_state["agent_feature_frame"]
    manifest = st.session_state["agent_feature_manifest"]
    feature_set = str(manifest.get("feature_set", "agent_feature_table"))
    source_file = str(manifest.get("path", "data/features"))

    st.subheader("Feature table preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(feature_frame):,}")
    col2.metric("Columns", f"{len(feature_frame.columns):,}")
    col3.metric("Manifest", manifest.get("run_id", "latest"))
    st.dataframe(make_streamlit_safe(feature_frame.head(100)), width="stretch")

    if st.button("Add feature table to modelling panel"):
        try:
            agent_series = feature_table_to_canonical_series(
                feature_frame,
                feature_set=feature_set,
                source_file=source_file,
            )
            if agent_series.empty:
                st.warning("No usable numeric feature series were found in this table.")
            else:
                merged = merge_series(st.session_state.get("series"), agent_series)
                st.session_state["series"] = merged
                st.session_state["catalogue"] = list_series_catalogue(merged)
                st.success(f"Added {agent_series['series_id'].nunique():,} agent feature series to the modelling panel.")
        except Exception as exc:
            st.error(f"Could not convert feature table into dashboard series: {exc}")

st.divider()
st.header("Normalized agent signals")
st.write(
    "This loads long-form normalized signal tables from `data/normalized/**/*.parquet`, "
    "for example UKHSA, wastewater, weather, or search-trend signals after the agent has "
    "mapped them into the shared respiratory schema."
)

if st.button("Load normalized agent signals"):
    try:
        normalized = cached_normalized_signals(str(ROOT))
        st.session_state["agent_normalized_signals"] = normalized
        if normalized.empty:
            st.warning("No normalized Parquet signal tables were found under `data/normalized/`.")
        else:
            st.success(f"Loaded {len(normalized):,} normalized signal rows.")
    except Exception as exc:
        st.error(f"Could not load normalized signals: {exc}")

if "agent_normalized_signals" in st.session_state and not st.session_state["agent_normalized_signals"].empty:
    normalized = st.session_state["agent_normalized_signals"]
    st.subheader("Normalized signal preview")
    st.dataframe(make_streamlit_safe(normalized.head(100)), width="stretch")

    if st.button("Add normalized signals to modelling panel"):
        try:
            signal_series = normalized_signals_to_canonical_series(normalized)
            if signal_series.empty:
                st.warning("No usable normalized signal series were found.")
            else:
                merged = merge_series(st.session_state.get("series"), signal_series)
                st.session_state["series"] = merged
                st.session_state["catalogue"] = list_series_catalogue(merged)
                st.success(f"Added {signal_series['series_id'].nunique():,} normalized signal series to the modelling panel.")
        except Exception as exc:
            st.error(f"Could not convert normalized signals into dashboard series: {exc}")

st.divider()
st.header("Current modelling panel")
if "series" in st.session_state:
    catalogue = st.session_state.get("catalogue")
    if catalogue is None:
        catalogue = list_series_catalogue(st.session_state["series"])
    st.success(f"The modelling panel contains {len(st.session_state['series']):,} observations across {len(catalogue):,} series.")
    st.dataframe(make_streamlit_safe(catalogue), width="stretch")
else:
    st.info("No modelling panel is loaded yet. Load canonical data on the Data page or add agent artifacts above.")

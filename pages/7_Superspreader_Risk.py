from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wastewater.superspreading import EventProfile, EpidemiologicalContext, assess_event_risk

st.set_page_config(page_title="Superspreader Risk", page_icon="⚠️", layout="wide")
st.title("Superspreader event risk")
st.write(
    "Estimate whether a candidate event could disproportionately amplify respiratory-virus "
    "transmission. The tool reports a Transmission Amplification Factor (TAF), expected event "
    "transmission, and the probability of exceeding a homogeneous superspreading threshold."
)

with st.expander("Metric definitions", expanded=False):
    st.markdown(
        """
- **TAF**: expected transmission under event conditions divided by expected transmission under baseline conditions, holding attendance, prevalence and R_t fixed.
- **Expected event transmission**: mean simulated secondary infections caused by infectious attendees at the event.
- **P(SSE)**: simulated probability that event-generated infections exceed the 99th-percentile homogeneous Poisson baseline with mean `infectious attendees × R_t`.
- **Regional contribution index**: expected event transmission divided by expected regional transmissions over the chosen window, if supplied.
"""
    )

st.sidebar.header("Event profile")
event_name = st.sidebar.text_input("Event name", value="Indoor concert")
event_id = event_name.strip().lower().replace(" ", "_") or "event"
pathogen = st.sidebar.selectbox("Pathogen", ["RSV", "influenza", "COVID-19", "other"])
geography_name = st.sidebar.text_input("Geography", value="England")
event_date = st.sidebar.date_input("Event date")
attendance = st.sidebar.number_input("Expected attendance", min_value=0, max_value=500_000, value=5_000, step=100)
duration_hours = st.sidebar.slider("Duration (hours)", min_value=0.25, max_value=24.0, value=3.0, step=0.25)
indoor = st.sidebar.checkbox("Indoor event", value=True)

col_a, col_b, col_c = st.columns(3)
with col_a:
    crowding_level = st.selectbox("Crowding", ["low", "medium", "high"], index=1)
    vocalisation_level = st.selectbox("Vocalisation / shouting / singing", ["low", "medium", "high"], index=1)
with col_b:
    ventilation_level = st.selectbox("Ventilation", ["good", "medium", "poor"], index=1)
    alcohol_level = st.selectbox("Alcohol / close social mixing", ["low", "medium", "high"], index=0)
with col_c:
    travel_mixing_level = st.selectbox("Travel catchment / mixing", ["low", "medium", "high"], index=1)
    mitigation_level = st.selectbox("Mitigation", ["low", "medium", "high"], index=0)

st.header("Epidemiological context")
st.write(
    "Use the best local estimate available from the main dashboard, UKHSA signals, wastewater, "
    "or recent model output. Prevalence is the fraction currently infectious, not cumulative incidence."
)

col1, col2, col3, col4 = st.columns(4)
prevalence_pct = col1.number_input("Infectious prevalence (%)", min_value=0.0, max_value=50.0, value=0.5, step=0.1)
rt = col2.number_input("Effective R_t", min_value=0.0, max_value=10.0, value=1.2, step=0.05)
attendance_prob = col3.number_input("P(infectious person attends)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
dispersion_k = col4.number_input("Dispersion k", min_value=0.01, max_value=5.0, value=0.2, step=0.01)

regional_expected = st.number_input(
    "Optional expected regional transmissions over same window",
    min_value=0.0,
    value=0.0,
    step=10.0,
)

n_sim = st.slider("Monte Carlo simulations", min_value=1_000, max_value=100_000, value=20_000, step=1_000)

if st.button("Assess superspreading risk", type="primary"):
    event = EventProfile(
        event_id=event_id,
        event_name=event_name,
        date=str(event_date),
        geography_name=geography_name,
        pathogen=pathogen,
        attendance=int(attendance),
        duration_hours=float(duration_hours),
        indoor=bool(indoor),
        crowding_level=crowding_level,
        ventilation_level=ventilation_level,
        vocalisation_level=vocalisation_level,
        alcohol_level=alcohol_level,
        travel_mixing_level=travel_mixing_level,
        mitigation_level=mitigation_level,
    )
    context = EpidemiologicalContext(
        prevalence=prevalence_pct / 100.0,
        reproduction_number=float(rt),
        attendance_probability_if_infectious=float(attendance_prob),
        dispersion_k=float(dispersion_k),
        regional_expected_transmissions=float(regional_expected) if regional_expected > 0 else None,
    )

    with st.spinner("Simulating event transmission..."):
        risk = assess_event_risk(event, context, n_sim=int(n_sim))

    st.session_state["last_superspreader_risk"] = risk.as_dict()

if "last_superspreader_risk" in st.session_state:
    risk = st.session_state["last_superspreader_risk"]
    st.header("Risk summary")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Risk band", str(risk["risk_band"]).upper())
    metric_cols[1].metric("TAF", f"{risk['transmission_amplification_factor']:.2f}×")
    metric_cols[2].metric("Expected secondary infections", f"{risk['expected_secondary_infections']:.1f}")
    metric_cols[3].metric("P(SSE)", f"{100 * risk['p_superspreading']:.1f}%")

    interval_cols = st.columns(4)
    interval_cols[0].metric("Expected infectious attendees", f"{risk['expected_infectious_attendees']:.2f}")
    interval_cols[1].metric("SSE threshold", f"{risk['superspreading_threshold']:.1f}")
    interval_cols[2].metric("Median secondary infections", f"{risk['secondary_infections_p50']:.1f}")
    if risk["regional_contribution_index"] is not None:
        interval_cols[3].metric("Regional contribution", f"{100 * risk['regional_contribution_index']:.2f}%")
    else:
        interval_cols[3].metric("Regional contribution", "not supplied")

    st.subheader("Uncertainty interval")
    st.write(
        f"Estimated secondary infections, 90% interval: "
        f"{risk['secondary_infections_p05']:.0f} to {risk['secondary_infections_p95']:.0f}."
    )
    st.progress(min(float(risk["p_superspreading"]), 1.0))

    st.subheader("Interpretation")
    st.write(risk["explanation"])

    st.subheader("Exportable result")
    result_frame = pd.DataFrame([risk])
    st.dataframe(result_frame, width="stretch")
    st.download_button(
        "Download result CSV",
        result_frame.to_csv(index=False).encode("utf-8"),
        file_name="superspreader_event_risk.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "This is an interpretable MVP intended for scenario comparison and planning. "
    "The coefficients and thresholds should be calibrated against outbreak investigations, "
    "venue data, and inferred regional event impacts before operational use."
)

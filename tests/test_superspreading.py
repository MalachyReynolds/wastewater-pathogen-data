from __future__ import annotations

import numpy as np

from wastewater.superspreading import EventProfile, EpidemiologicalContext, assess_event_risk
from wastewater.superspreading.mechanistic import transmission_amplification_factor
from wastewater.superspreading.simulation import simulate_event_transmission


def make_event(**overrides) -> EventProfile:
    values = {
        "event_id": "test_event",
        "event_name": "Test Event",
        "date": "2026-01-01",
        "geography_name": "England",
        "pathogen": "influenza",
        "attendance": 5_000,
        "duration_hours": 3.0,
        "indoor": True,
        "crowding_level": "medium",
        "ventilation_level": "medium",
        "vocalisation_level": "medium",
        "alcohol_level": "low",
        "travel_mixing_level": "medium",
        "mitigation_level": "low",
    }
    values.update(overrides)
    return EventProfile(**values)


def test_transmission_amplification_higher_for_risky_event() -> None:
    low_risk = make_event(
        indoor=False,
        crowding_level="low",
        ventilation_level="good",
        vocalisation_level="low",
        mitigation_level="high",
    )
    high_risk = make_event(
        indoor=True,
        crowding_level="high",
        ventilation_level="poor",
        vocalisation_level="high",
        alcohol_level="high",
        travel_mixing_level="high",
        mitigation_level="low",
        duration_hours=6.0,
    )

    assert transmission_amplification_factor(high_risk) > transmission_amplification_factor(low_risk)
    assert transmission_amplification_factor(low_risk) < 1.0


def test_simulation_returns_zero_when_reproduction_number_zero() -> None:
    samples = simulate_event_transmission(
        attendance=1_000,
        prevalence=0.01,
        attendance_probability_if_infectious=0.7,
        reproduction_number=0.0,
        transmission_amplification_factor=5.0,
        dispersion_k=0.2,
        n_sim=1_000,
    )
    assert np.all(samples == 0)


def test_assess_event_risk_outputs_expected_fields() -> None:
    event = make_event()
    context = EpidemiologicalContext(prevalence=0.005, reproduction_number=1.2, dispersion_k=0.2)

    risk = assess_event_risk(event, context, n_sim=2_000, random_state=123)

    assert risk.event_id == event.event_id
    assert risk.expected_infectious_attendees > 0
    assert risk.transmission_amplification_factor > 0
    assert risk.expected_secondary_infections >= 0
    assert 0 <= risk.p_superspreading <= 1
    assert risk.secondary_infections_p05 <= risk.secondary_infections_p50 <= risk.secondary_infections_p95
    assert risk.risk_band in {"low", "moderate", "high", "very high"}


def test_high_risk_event_has_larger_expected_transmission_than_low_risk_event() -> None:
    context = EpidemiologicalContext(prevalence=0.01, reproduction_number=1.3, dispersion_k=0.2)
    low = make_event(
        indoor=False,
        crowding_level="low",
        ventilation_level="good",
        vocalisation_level="low",
        mitigation_level="high",
    )
    high = make_event(
        indoor=True,
        crowding_level="high",
        ventilation_level="poor",
        vocalisation_level="high",
        alcohol_level="high",
        travel_mixing_level="high",
        mitigation_level="low",
        duration_hours=6.0,
    )

    low_risk = assess_event_risk(low, context, n_sim=2_000, random_state=42)
    high_risk = assess_event_risk(high, context, n_sim=2_000, random_state=42)

    assert high_risk.transmission_amplification_factor > low_risk.transmission_amplification_factor
    assert high_risk.expected_secondary_infections > low_risk.expected_secondary_infections

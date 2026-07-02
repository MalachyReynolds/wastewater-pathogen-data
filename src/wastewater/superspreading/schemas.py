from __future__ import annotations

from dataclasses import dataclass


Level = str


@dataclass(frozen=True)
class EventProfile:
    """Description of a candidate event or gathering.

    Categorical levels should usually be one of ``low``, ``medium`` or ``high``;
    ventilation can additionally be ``good`` or ``poor``. The model is tolerant
    of unknown values and treats them as neutral.
    """

    event_id: str
    event_name: str
    date: str
    geography_name: str
    pathogen: str
    attendance: int
    duration_hours: float
    indoor: bool
    crowding_level: Level = "medium"
    ventilation_level: Level = "medium"
    vocalisation_level: Level = "medium"
    alcohol_level: Level = "low"
    travel_mixing_level: Level = "medium"
    mitigation_level: Level = "low"


@dataclass(frozen=True)
class EpidemiologicalContext:
    """Local epidemiological context at or near event time."""

    prevalence: float
    reproduction_number: float
    attendance_probability_if_infectious: float = 0.7
    dispersion_k: float = 0.2
    regional_expected_transmissions: float | None = None


@dataclass(frozen=True)
class SuperspreadingRisk:
    """Risk and amplification summary for a candidate event."""

    event_id: str
    event_name: str
    pathogen: str
    expected_infectious_attendees: float
    transmission_amplification_factor: float
    expected_secondary_infections: float
    superspreading_threshold: float
    p_superspreading: float
    secondary_infections_p05: float
    secondary_infections_p50: float
    secondary_infections_p95: float
    risk_band: str
    regional_contribution_index: float | None
    explanation: str

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "event_id": self.event_id,
            "event_name": self.event_name,
            "pathogen": self.pathogen,
            "expected_infectious_attendees": self.expected_infectious_attendees,
            "transmission_amplification_factor": self.transmission_amplification_factor,
            "expected_secondary_infections": self.expected_secondary_infections,
            "superspreading_threshold": self.superspreading_threshold,
            "p_superspreading": self.p_superspreading,
            "secondary_infections_p05": self.secondary_infections_p05,
            "secondary_infections_p50": self.secondary_infections_p50,
            "secondary_infections_p95": self.secondary_infections_p95,
            "risk_band": self.risk_band,
            "regional_contribution_index": self.regional_contribution_index,
            "explanation": self.explanation,
        }

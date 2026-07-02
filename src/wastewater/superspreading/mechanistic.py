from __future__ import annotations

import math

from .schemas import EventProfile


LEVEL_WEIGHTS = {
    "crowding": {
        "low": -0.15,
        "medium": 0.25,
        "high": 0.70,
    },
    "ventilation": {
        "good": -0.45,
        "medium": 0.20,
        "poor": 0.80,
        "low": 0.80,
        "high": -0.45,
    },
    "vocalisation": {
        "low": 0.0,
        "medium": 0.30,
        "high": 0.70,
    },
    "alcohol": {
        "low": 0.0,
        "medium": 0.15,
        "high": 0.35,
    },
    "travel_mixing": {
        "low": 0.0,
        "medium": 0.20,
        "high": 0.45,
    },
    "mitigation": {
        "low": 0.0,
        "medium": -0.35,
        "high": -0.70,
    },
}


def _level_weight(kind: str, value: str) -> float:
    return LEVEL_WEIGHTS.get(kind, {}).get(str(value).strip().lower(), 0.0)


def event_log_amplification(event: EventProfile) -> float:
    """Return log transmission amplification for event conditions.

    The coefficients are intentionally transparent defaults for the MVP. They
    should be calibrated later against outbreak investigations or inferred event
    impacts. Positive values increase event-level transmission efficiency and
    negative values reduce it relative to a baseline gathering.
    """
    score = 0.0
    score += 0.60 if event.indoor else -0.25
    score += _level_weight("crowding", event.crowding_level)
    score += _level_weight("ventilation", event.ventilation_level)
    score += _level_weight("vocalisation", event.vocalisation_level)
    score += _level_weight("alcohol", event.alcohol_level)
    score += _level_weight("travel_mixing", event.travel_mixing_level)
    score += _level_weight("mitigation", event.mitigation_level)

    if event.duration_hours > 8:
        score += 0.60
    elif event.duration_hours > 4:
        score += 0.30
    elif event.duration_hours < 1:
        score -= 0.15

    return score


def transmission_amplification_factor(event: EventProfile, *, cap: float = 25.0) -> float:
    """Return the Transmission Amplification Factor (TAF).

    TAF is defined as expected secondary infections under event conditions
    divided by expected secondary infections under baseline conditions, holding
    local prevalence, attendance and reproduction number fixed.
    """
    return min(float(math.exp(event_log_amplification(event))), cap)


def factor_contributions(event: EventProfile) -> dict[str, float]:
    """Break down additive log-scale contributions used in the TAF score."""
    duration = 0.0
    if event.duration_hours > 8:
        duration = 0.60
    elif event.duration_hours > 4:
        duration = 0.30
    elif event.duration_hours < 1:
        duration = -0.15
    return {
        "indoor": 0.60 if event.indoor else -0.25,
        "crowding": _level_weight("crowding", event.crowding_level),
        "ventilation": _level_weight("ventilation", event.ventilation_level),
        "vocalisation": _level_weight("vocalisation", event.vocalisation_level),
        "alcohol": _level_weight("alcohol", event.alcohol_level),
        "travel_mixing": _level_weight("travel_mixing", event.travel_mixing_level),
        "mitigation": _level_weight("mitigation", event.mitigation_level),
        "duration": duration,
    }

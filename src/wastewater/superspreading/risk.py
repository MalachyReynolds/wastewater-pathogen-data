from __future__ import annotations

import numpy as np

from .mechanistic import factor_contributions, transmission_amplification_factor
from .schemas import EpidemiologicalContext, EventProfile, SuperspreadingRisk
from .simulation import poisson_quantile_approx, simulate_event_transmission


def expected_infectious_attendees(event: EventProfile, context: EpidemiologicalContext) -> float:
    return event.attendance * context.prevalence * context.attendance_probability_if_infectious


def risk_band(p_superspreading: float, expected_secondary_infections: float) -> str:
    """Map probability and expected impact to an interpretable risk band."""
    if p_superspreading >= 0.35 or expected_secondary_infections >= 50:
        return "very high"
    if p_superspreading >= 0.15 or expected_secondary_infections >= 20:
        return "high"
    if p_superspreading >= 0.05 or expected_secondary_infections >= 5:
        return "moderate"
    return "low"


def build_explanation(event: EventProfile, context: EpidemiologicalContext, taf: float) -> str:
    contributions = factor_contributions(event)
    positive = sorted(((k, v) for k, v in contributions.items() if v > 0), key=lambda item: item[1], reverse=True)
    negative = sorted(((k, v) for k, v in contributions.items() if v < 0), key=lambda item: item[1])
    drivers = ", ".join(name.replace("_", " ") for name, _ in positive[:3]) or "baseline event conditions"
    mitigators = ", ".join(name.replace("_", " ") for name, _ in negative[:2]) or "none"
    return (
        f"TAF is {taf:.2f}. Main amplifying factors: {drivers}. "
        f"Main reducing factors: {mitigators}. Context uses prevalence={context.prevalence:.4f}, "
        f"R_t={context.reproduction_number:.2f}, dispersion k={context.dispersion_k:.2f}."
    )


def assess_event_risk(
    event: EventProfile,
    context: EpidemiologicalContext,
    *,
    n_sim: int = 20_000,
    random_state: int | None = 42,
) -> SuperspreadingRisk:
    """Estimate event-level superspreading risk and amplification.

    Metrics:
    - TAF: expected transmission under event conditions divided by baseline.
    - EET: expected event transmission, i.e. mean simulated secondary infections.
    - P_SSE: probability simulated transmission exceeds a homogeneous Poisson
      99th-percentile baseline with mean I_e * R_t.
    """
    taf = transmission_amplification_factor(event)
    infectious = expected_infectious_attendees(event, context)
    baseline_mean = infectious * context.reproduction_number
    threshold = poisson_quantile_approx(baseline_mean, quantile=0.99)

    samples = simulate_event_transmission(
        attendance=event.attendance,
        prevalence=context.prevalence,
        attendance_probability_if_infectious=context.attendance_probability_if_infectious,
        reproduction_number=context.reproduction_number,
        transmission_amplification_factor=taf,
        dispersion_k=context.dispersion_k,
        n_sim=n_sim,
        random_state=random_state,
    )

    expected_secondary = float(np.mean(samples))
    p_sse = float(np.mean(samples > threshold))
    p05, p50, p95 = (float(value) for value in np.quantile(samples, [0.05, 0.50, 0.95]))
    regional_contribution = None
    if context.regional_expected_transmissions and context.regional_expected_transmissions > 0:
        regional_contribution = expected_secondary / context.regional_expected_transmissions

    return SuperspreadingRisk(
        event_id=event.event_id,
        event_name=event.event_name,
        pathogen=event.pathogen,
        expected_infectious_attendees=float(infectious),
        transmission_amplification_factor=float(taf),
        expected_secondary_infections=expected_secondary,
        superspreading_threshold=float(threshold),
        p_superspreading=p_sse,
        secondary_infections_p05=p05,
        secondary_infections_p50=p50,
        secondary_infections_p95=p95,
        risk_band=risk_band(p_sse, expected_secondary),
        regional_contribution_index=regional_contribution,
        explanation=build_explanation(event, context, taf),
    )

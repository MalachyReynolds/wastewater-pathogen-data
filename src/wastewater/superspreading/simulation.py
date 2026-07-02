from __future__ import annotations

import numpy as np


def simulate_event_transmission(
    *,
    attendance: int,
    prevalence: float,
    attendance_probability_if_infectious: float,
    reproduction_number: float,
    transmission_amplification_factor: float,
    dispersion_k: float,
    n_sim: int = 20_000,
    random_state: int | None = 42,
) -> np.ndarray:
    """Simulate event-generated secondary infections.

    The model first draws the number of infectious attendees, then draws each
    infectious attendee's secondary infections from a negative-binomial offspring
    distribution with mean ``R_t * TAF`` and dispersion ``k``.
    """
    if attendance < 0:
        raise ValueError("attendance must be non-negative")
    if not 0 <= prevalence <= 1:
        raise ValueError("prevalence must be between 0 and 1")
    if not 0 <= attendance_probability_if_infectious <= 1:
        raise ValueError("attendance_probability_if_infectious must be between 0 and 1")
    if reproduction_number < 0:
        raise ValueError("reproduction_number must be non-negative")
    if transmission_amplification_factor < 0:
        raise ValueError("transmission_amplification_factor must be non-negative")
    if dispersion_k <= 0:
        raise ValueError("dispersion_k must be positive")
    if n_sim <= 0:
        raise ValueError("n_sim must be positive")

    rng = np.random.default_rng(random_state)
    infectious_probability = min(prevalence * attendance_probability_if_infectious, 1.0)
    infectious = rng.binomial(attendance, infectious_probability, size=n_sim)

    mean_secondary = reproduction_number * transmission_amplification_factor
    if mean_secondary <= 0:
        return np.zeros(n_sim, dtype=float)

    # NumPy's negative_binomial(n, p) has mean n * (1-p) / p. Setting
    # p = k / (k + mu) gives mean mu and variance mu + mu^2 / k.
    k = float(dispersion_k)
    p = k / (k + mean_secondary)
    totals = np.zeros(n_sim, dtype=float)
    unique_counts, frequencies = np.unique(infectious, return_counts=True)
    for n_infectious, freq in zip(unique_counts, frequencies):
        if n_infectious == 0:
            continue
        draws = rng.negative_binomial(k, p, size=(freq, int(n_infectious)))
        totals[infectious == n_infectious] = draws.sum(axis=1)
    return totals


def poisson_quantile_approx(mean: float, quantile: float = 0.99) -> float:
    """Approximate a high Poisson quantile without requiring SciPy.

    This uses a normal approximation with continuity correction. It is adequate
    for an MVP risk threshold; calibration can replace it later.
    """
    if mean <= 0:
        return 1.0
    # z for 0.99. Keep explicit to avoid an additional dependency.
    z = 2.3263478740408408 if quantile == 0.99 else 1.6448536269514722
    return max(1.0, mean + z * np.sqrt(mean) + 0.5)

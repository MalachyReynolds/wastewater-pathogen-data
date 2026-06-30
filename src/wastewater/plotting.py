"""Plotting helpers for wastewater analysis notebooks."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_country_pathogen_timeseries(
    df: pd.DataFrame,
    pathogen: str,
    value_col: str = "zscore_within_series",
    date_col: str = "week",
    country_col: str = "country",
    figsize: tuple[int, int] = (12, 5),
):
    """Plot country-level mean time series for one pathogen."""
    subset = df[df["pathogen"].astype(str).str.lower() == pathogen.lower()].copy()
    if subset.empty:
        raise ValueError(f"No rows found for pathogen={pathogen!r}")

    weekly = (
        subset.groupby([country_col, date_col], dropna=False)[value_col]
        .mean()
        .reset_index()
        .sort_values(date_col)
    )

    fig, ax = plt.subplots(figsize=figsize)
    for country, group in weekly.groupby(country_col):
        ax.plot(group[date_col], group[value_col], label=country)

    ax.set_title(f"Wastewater signal: {pathogen}")
    ax.set_xlabel("Week")
    ax.set_ylabel(value_col)
    ax.legend()
    fig.autofmt_xdate()
    return fig, ax


def plot_coverage(coverage: pd.DataFrame, figsize: tuple[int, int] = (10, 5)):
    """Plot observation counts by country/pathogen from a coverage table."""
    pivot = coverage.pivot_table(
        index="country",
        columns="pathogen",
        values="n_observations",
        aggfunc="sum",
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=figsize)
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Available observations by country and pathogen")
    ax.set_ylabel("Observations")
    fig.tight_layout()
    return fig, ax

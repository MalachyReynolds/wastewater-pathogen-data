from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"


@dataclass(frozen=True)
class AgentConfig:
    """Credentials and model selection for calling Nebius Token Factory."""

    api_key: str
    model: str
    base_url: str = DEFAULT_BASE_URL


def load_config() -> AgentConfig:
    """Read agent configuration from the environment.

    ``NEBIUS_API_KEY`` and ``NEBIUS_MODEL`` are required and never hardcoded --
    Nebius Token Factory hosts many models and the available ones can vary by
    account, so a wrong hardcoded default would silently fail or bill the
    wrong model. ``NEBIUS_BASE_URL`` is optional and defaults to Token
    Factory's published OpenAI-compatible endpoint.
    """
    api_key = os.getenv("NEBIUS_API_KEY")
    model = os.getenv("NEBIUS_MODEL")
    missing = [name for name, value in [("NEBIUS_API_KEY", api_key), ("NEBIUS_MODEL", model)] if not value]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Set them before running the data agent, e.g. "
            "NEBIUS_API_KEY=... NEBIUS_MODEL=... python scripts/run_data_agent.py"
        )
    base_url = os.getenv("NEBIUS_BASE_URL", DEFAULT_BASE_URL)
    return AgentConfig(api_key=api_key, model=model, base_url=base_url)

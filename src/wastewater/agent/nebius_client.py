from __future__ import annotations

from openai import OpenAI

from .config import AgentConfig, load_config


def forge_nebius_client(config: AgentConfig | None = None) -> OpenAI:
    """Construct an OpenAI-SDK client pointed at Nebius Token Factory.

    Nebius Token Factory exposes an OpenAI-compatible chat completions API, so
    the standard ``openai`` SDK works unmodified against it via ``base_url``.
    This is the "token forge": it turns the ``NEBIUS_API_KEY`` environment
    variable into a ready-to-use, authenticated client -- nothing more exotic
    than that.
    """
    if config is None:
        config = load_config()
    return OpenAI(api_key=config.api_key, base_url=config.base_url)

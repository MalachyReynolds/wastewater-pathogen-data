from __future__ import annotations

import pytest

from wastewater.agent.config import DEFAULT_BASE_URL, AgentConfig, load_config
from wastewater.agent.nebius_client import forge_nebius_client


def test_load_config_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.setenv("NEBIUS_MODEL", "some-model")
    with pytest.raises(RuntimeError, match="NEBIUS_API_KEY"):
        load_config()


def test_load_config_raises_when_model_missing(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.delenv("NEBIUS_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="NEBIUS_MODEL"):
        load_config()


def test_load_config_reads_env_vars(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("NEBIUS_MODEL", "test-model")
    monkeypatch.delenv("NEBIUS_BASE_URL", raising=False)

    config = load_config()

    assert config.api_key == "test-key"
    assert config.model == "test-model"
    assert config.base_url == DEFAULT_BASE_URL


def test_load_config_honours_custom_base_url(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("NEBIUS_MODEL", "test-model")
    monkeypatch.setenv("NEBIUS_BASE_URL", "https://example.invalid/v1/")

    config = load_config()

    assert config.base_url == "https://example.invalid/v1/"


def test_forge_nebius_client_uses_config():
    config = AgentConfig(api_key="test-key", model="test-model", base_url="https://example.invalid/v1/")
    client = forge_nebius_client(config)

    assert client.api_key == "test-key"
    assert str(client.base_url) == "https://example.invalid/v1/"

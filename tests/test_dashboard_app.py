from __future__ import annotations

import importlib


def test_dashboard_entrypoint_imports():
    module = importlib.import_module("app")
    assert module is not None

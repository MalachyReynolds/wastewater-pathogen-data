from __future__ import annotations

from wastewater.agent.llm_tasks import flag_anomalies, infer_column_mapping, summarize_manifest


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletionResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content=None, exc=None):
        self._content = content
        self._exc = exc

    def create(self, **kwargs):
        if self._exc is not None:
            raise self._exc
        return _FakeCompletionResponse(self._content)


class _FakeChat:
    def __init__(self, content=None, exc=None):
        self.completions = _FakeCompletions(content=content, exc=exc)


class FakeClient:
    def __init__(self, content=None, exc=None):
        self.chat = _FakeChat(content=content, exc=exc)


def test_infer_column_mapping_uses_llm_response_when_valid():
    columns = ["date", "location", "new_cases", "new_deaths"]
    content = '{"date_column": "date", "geography_column": "location", "signal_columns": ["new_cases", "new_deaths"]}'
    client = FakeClient(content=content)

    mapping = infer_column_mapping(client, "test-model", columns, sample_rows=[{"date": "2026-01-01"}])

    assert mapping == {
        "date_column": "date",
        "geography_column": "location",
        "signal_columns": ["new_cases", "new_deaths"],
    }


def test_infer_column_mapping_falls_back_on_malformed_json():
    columns = ["date", "location", "new_cases"]
    client = FakeClient(content="not json at all")

    mapping = infer_column_mapping(client, "test-model", columns, sample_rows=[])

    assert mapping["date_column"] == "date"
    assert mapping["geography_column"] == "location"
    assert mapping["signal_columns"] == ["new_cases"]


def test_infer_column_mapping_falls_back_on_client_exception():
    columns = ["date", "new_cases"]
    client = FakeClient(exc=RuntimeError("network down"))

    mapping = infer_column_mapping(client, "test-model", columns, sample_rows=[])

    assert mapping["date_column"] == "date"
    assert mapping["signal_columns"] == ["new_cases"]


def test_infer_column_mapping_falls_back_when_llm_invents_unknown_column():
    columns = ["date", "new_cases"]
    content = '{"date_column": "date", "geography_column": null, "signal_columns": ["made_up_column"]}'
    client = FakeClient(content=content)

    mapping = infer_column_mapping(client, "test-model", columns, sample_rows=[])

    assert mapping["signal_columns"] == ["new_cases"]


def test_summarize_manifest_uses_llm_text_when_available():
    client = FakeClient(content="  A tidy one-sentence summary.  ")

    summary = summarize_manifest(client, "test-model", "owid_covid", {"rows": 100})

    assert summary == "A tidy one-sentence summary."


def test_summarize_manifest_falls_back_on_exception():
    client = FakeClient(exc=RuntimeError("boom"))

    summary = summarize_manifest(client, "test-model", "owid_covid", {"rows": 100, "signal_count": 3})

    assert "owid_covid" in summary
    assert "100" in summary


def test_flag_anomalies_uses_llm_response_when_valid():
    client = FakeClient(content='{"validation_status": "warning", "notes": "some nulls"}')

    result = flag_anomalies(client, "test-model", {"missing_value_fraction": 0.05})

    assert result == {"validation_status": "warning", "notes": "some nulls"}


def test_flag_anomalies_falls_back_on_invalid_status():
    client = FakeClient(content='{"validation_status": "definitely_fine", "notes": "x"}')

    result = flag_anomalies(client, "test-model", {"missing_value_fraction": 0.0})

    assert result["validation_status"] == "passed"


def test_flag_anomalies_falls_back_on_exception_using_heuristic():
    client = FakeClient(exc=RuntimeError("boom"))

    result = flag_anomalies(client, "test-model", {"missing_value_fraction": 0.6})

    assert result["validation_status"] == "failed"

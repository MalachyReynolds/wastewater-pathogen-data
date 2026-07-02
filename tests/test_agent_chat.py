from __future__ import annotations

import json
from pathlib import Path

import wastewater.agent.tools as agent_tools
from wastewater.agent.chat import execute_confirmed_action, run_chat_turn
from wastewater.agent.sources import load_custom_sources


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id_, name, arguments):
        self.id = id_
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        return self._responses.pop(0)


class _FakeChat:
    def __init__(self, responses):
        self.completions = _FakeCompletions(responses)


class ScriptedFakeClient:
    """A stub OpenAI-style client returning a queued sequence of responses, one per call."""

    def __init__(self, responses):
        self.chat = _FakeChat(responses)


def test_run_chat_turn_executes_read_only_tool_then_returns_final_text(monkeypatch):
    monkeypatch.setattr(
        agent_tools,
        "search_catalog",
        lambda query, searcher=None: [{"title": "Weekly flu admissions", "slug": "flu-slug", "type": "chart"}],
    )

    responses = [
        _FakeResponse(
            _FakeMessage(
                tool_calls=[_FakeToolCall("call_1", "search_catalog", json.dumps({"query": "influenza"}))]
            )
        ),
        _FakeResponse(_FakeMessage(content="I found a dataset called 'Weekly flu admissions'.")),
    ]
    client = ScriptedFakeClient(responses)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "find flu data"}]

    result = run_chat_turn(client, "test-model", messages, context={}, known_source_names=[])

    assert result.pending_action is None
    assert result.messages[-1]["role"] == "assistant"
    assert result.messages[-1]["content"] == "I found a dataset called 'Weekly flu admissions'."
    tool_messages = [message for message in result.messages if message["role"] == "tool"]
    assert len(tool_messages) == 1
    assert json.loads(tool_messages[0]["content"]) == [{"title": "Weekly flu admissions", "slug": "flu-slug", "type": "chart"}]


def test_run_chat_turn_stops_at_pending_action_without_a_second_llm_call():
    arguments = json.dumps(
        {
            "name": "flu_hosp",
            "pathogen": "influenza",
            "role": "predicted",
            "description": "test source",
            "catalog_slug": "flu-slug",
        }
    )
    responses = [
        _FakeResponse(_FakeMessage(tool_calls=[_FakeToolCall("call_1", "propose_add_source", arguments)])),
        # A second response that should never be consumed -- proves the loop stops.
        _FakeResponse(_FakeMessage(content="this should not be reached")),
    ]
    client = ScriptedFakeClient(responses)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "add flu hospital data as a source"}]

    result = run_chat_turn(client, "test-model", messages, context={}, known_source_names=[])

    assert result.pending_action is not None
    assert result.pending_action["name"] == "propose_add_source"
    assert result.pending_action["proposal"]["name"] == "flu_hosp"
    assert result.pending_action["proposal"]["role"] == "predicted"
    assert (
        result.pending_action["summary"]
        == "Add a new source 'flu_hosp' (influenza, role: predicted) from catalog slug 'flu-slug'?"
    )
    # the second scripted response must not have been consumed
    assert len(client.chat.completions._responses) == 1


def test_run_chat_turn_invalid_proposal_does_not_set_pending_action_and_continues():
    bad_arguments = json.dumps(
        {"name": "x", "pathogen": "flu", "role": "predictive", "description": "test"}
    )  # missing url/catalog_slug
    responses = [
        _FakeResponse(_FakeMessage(tool_calls=[_FakeToolCall("call_1", "propose_add_source", bad_arguments)])),
        _FakeResponse(_FakeMessage(content="I need either a URL or a catalog slug -- which do you have?")),
    ]
    client = ScriptedFakeClient(responses)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "add a source called x"}]

    result = run_chat_turn(client, "test-model", messages, context={}, known_source_names=[])

    assert result.pending_action is None
    assert result.messages[-1]["content"] == "I need either a URL or a catalog slug -- which do you have?"


def test_run_chat_turn_executes_search_google_trends_tool(monkeypatch):
    monkeypatch.setattr(
        agent_tools,
        "search_google_trends",
        lambda query, searcher=None: [{"title": "Flu", "term": "/m/0cycc", "type": "Disease"}],
    )
    responses = [
        _FakeResponse(
            _FakeMessage(tool_calls=[_FakeToolCall("call_1", "search_google_trends", json.dumps({"query": "influenza"}))])
        ),
        _FakeResponse(_FakeMessage(content="I found a Google Trends topic called 'Flu'.")),
    ]
    client = ScriptedFakeClient(responses)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "search google trends for influenza"}]

    result = run_chat_turn(client, "test-model", messages, context={}, known_source_names=[])

    assert result.pending_action is None
    tool_messages = [message for message in result.messages if message["role"] == "tool"]
    assert json.loads(tool_messages[0]["content"]) == [{"title": "Flu", "term": "/m/0cycc", "type": "Disease"}]


def test_run_chat_turn_threads_root_through_to_local_google_trends_search(tmp_path: Path):
    trends_dir = tmp_path / "Google_trends_v2" / "1y_data"
    trends_dir.mkdir(parents=True)
    (trends_dir / "time_series_GB_test.csv").write_text('"Time","cough"\n"2026-01-04",37\n', encoding="utf-8")

    responses = [
        _FakeResponse(
            _FakeMessage(
                tool_calls=[_FakeToolCall("call_1", "search_local_google_trends_files", json.dumps({"query": "cough"}))]
            )
        ),
        _FakeResponse(_FakeMessage(content="Found a local file for 'cough'.")),
    ]
    client = ScriptedFakeClient(responses)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "search local google trends files for cough"}]

    result = run_chat_turn(client, "test-model", messages, context={}, known_source_names=[], root=tmp_path)

    tool_messages = [message for message in result.messages if message["role"] == "tool"]
    assert json.loads(tool_messages[0]["content"]) == [
        {"term": "cough", "local_file": "Google_trends_v2/1y_data/time_series_GB_test.csv", "period": "1y"}
    ]


def test_run_chat_turn_pending_action_summary_for_google_trends_term():
    arguments = json.dumps(
        {
            "name": "flu_trends",
            "pathogen": "influenza",
            "role": "predictive",
            "description": "test source",
            "google_trends_term": "/m/0cycc",
            "google_trends_geo": "GB",
            "google_trends_timeframe": "today 5-y",
        }
    )
    responses = [_FakeResponse(_FakeMessage(tool_calls=[_FakeToolCall("call_1", "propose_add_source", arguments)]))]
    client = ScriptedFakeClient(responses)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "add flu trends as a source"}]

    result = run_chat_turn(client, "test-model", messages, context={}, known_source_names=[])

    assert result.pending_action["proposal"]["google_trends_term"] == "/m/0cycc"
    assert (
        result.pending_action["summary"]
        == "Add a new source 'flu_trends' (influenza, role: predictive) from Google Trends term '/m/0cycc' (geo=GB, today 5-y)?"
    )


def test_run_chat_turn_pending_action_summary_for_google_trends_local_file():
    arguments = json.dumps(
        {
            "name": "cough_trends",
            "pathogen": "respiratory",
            "role": "predictive",
            "description": "test source",
            "google_trends_local_file": "Google_trends_v2/1y_data/time_series_GB_a.csv",
        }
    )
    responses = [_FakeResponse(_FakeMessage(tool_calls=[_FakeToolCall("call_1", "propose_add_source", arguments)]))]
    client = ScriptedFakeClient(responses)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "add the cough local file as a source"}]

    result = run_chat_turn(client, "test-model", messages, context={}, known_source_names=[])

    assert (
        result.pending_action["summary"]
        == "Add a new source 'cough_trends' (respiratory, role: predictive) "
        "from local file 'Google_trends_v2/1y_data/time_series_GB_a.csv'?"
    )


def test_execute_confirmed_action_persists_google_trends_source(tmp_path: Path):
    pending_action = {
        "name": "propose_add_source",
        "proposal": {
            "name": "flu_trends",
            "pathogen": "influenza",
            "description": "test source",
            "role": "predictive",
            "url": None,
            "catalog_slug": None,
            "google_trends_term": "/m/0cycc",
            "google_trends_geo": "US",
            "google_trends_timeframe": "today 12-m",
            "google_trends_local_file": None,
        },
    }

    result_text = execute_confirmed_action(pending_action, tmp_path, client=None, model="test-model")

    assert "flu_trends" in result_text
    sources = load_custom_sources(tmp_path)
    assert len(sources) == 1
    assert sources[0].google_trends_term == "/m/0cycc"
    assert sources[0].google_trends_geo == "US"
    assert sources[0].google_trends_timeframe == "today 12-m"

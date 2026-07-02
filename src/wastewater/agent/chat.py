"""Chat orchestration for the dashboard's "Chat with the agent" section.

The model can call read-only tools freely; the moment it calls a mutating tool
(``propose_add_source``/``propose_run_ingestion``), the turn stops and control
returns to the UI so the user can confirm or cancel -- the model never causes
a write or a paid ingestion run on its own. See ``tools.py`` for why.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import ingest as agent_ingest
from . import tools as agent_tools
from .sources import SourceSpec, add_custom_source, list_sources

SYSTEM_PROMPT = (
    "You are the respiratory-data ingestion agent for this dashboard. You can search the OWID "
    "data catalog, search Google Trends (live, via search_google_trends) or the Google Trends "
    "CSV exports already saved in this repository (via search_local_google_trends_files), "
    "report on what's currently loaded in the dashboard, and propose adding a new source or "
    "running ingestion for one. Adding a source or running ingestion always requires the user's "
    "explicit confirmation in the dashboard UI -- when you call propose_add_source or "
    "propose_run_ingestion, you are only proposing the action, not performing it. Never tell the "
    "user an action has been completed unless a tool result says so. Keep responses concise."
)

READ_ONLY_TOOLS = {"search_catalog", "search_google_trends", "search_local_google_trends_files", "get_dashboard_status"}
PROPOSAL_TOOLS = {"propose_add_source", "propose_run_ingestion"}
MAX_TOOL_ITERATIONS = 5


@dataclass
class ChatTurnResult:
    messages: list[dict[str, Any]]
    pending_action: dict[str, Any] | None = None


def _describe_proposal(name: str, arguments: dict[str, Any], proposal: dict[str, Any]) -> str:
    """Build the confirmation text shown to the user, deterministically from the validated
    proposal -- not from the model's own words, so what's confirmed is exactly what runs."""
    if name == "propose_add_source":
        if proposal.get("catalog_slug"):
            location = f"catalog slug '{proposal['catalog_slug']}'"
        elif proposal.get("google_trends_term"):
            location = (
                f"Google Trends term '{proposal['google_trends_term']}' "
                f"(geo={proposal['google_trends_geo'] or 'worldwide'}, {proposal['google_trends_timeframe']})"
            )
        elif proposal.get("google_trends_local_file"):
            location = f"local file '{proposal['google_trends_local_file']}'"
        else:
            location = f"URL {proposal['url']}"
        return f"Add a new source '{proposal['name']}' ({proposal['pathogen']}, role: {proposal['role']}) from {location}?"
    if name == "propose_run_ingestion":
        return f"Run ingestion for source '{proposal['source_name']}'? This calls the LLM and writes new files."
    return f"Perform action '{name}' with arguments {arguments}?"


def _execute_read_only_tool(name: str, arguments: dict[str, Any], context: dict[str, Any], root: Path) -> Any:
    if name == "search_catalog":
        return agent_tools.search_catalog(arguments["query"])
    if name == "search_google_trends":
        return agent_tools.search_google_trends(arguments["query"])
    if name == "search_local_google_trends_files":
        return agent_tools.search_local_google_trends_files(arguments["query"], root)
    if name == "get_dashboard_status":
        return agent_tools.get_dashboard_status(context)
    raise ValueError(f"Not a read-only tool: {name}")


def _validate_proposal_tool(name: str, arguments: dict[str, Any], known_source_names: list[str]) -> dict[str, Any]:
    if name == "propose_add_source":
        return agent_tools.propose_add_source(**arguments)
    if name == "propose_run_ingestion":
        return agent_tools.propose_run_ingestion(arguments["source_name"], known_source_names)
    raise ValueError(f"Not a proposal tool: {name}")


def run_chat_turn(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    context: dict[str, Any],
    known_source_names: list[str],
    root: Path = Path("."),
) -> ChatTurnResult:
    """Run one user turn: call the LLM, execute read-only tools automatically, and stop for
    confirmation the moment a mutating tool is proposed."""
    messages = list(messages)

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=agent_tools.TOOL_SCHEMAS,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message

        message_dict: dict[str, Any] = {"role": "assistant", "content": assistant_message.content}
        tool_calls = assistant_message.tool_calls or []
        if tool_calls:
            message_dict["tool_calls"] = [
                {"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}}
                for call in tool_calls
            ]
        messages.append(message_dict)

        if not tool_calls:
            return ChatTurnResult(messages=messages, pending_action=None)

        pending_action = None
        for tool_call in tool_calls:
            name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            if name in READ_ONLY_TOOLS:
                result = _execute_read_only_tool(name, arguments, context, root)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)})
            elif name in PROPOSAL_TOOLS:
                proposal = _validate_proposal_tool(name, arguments, known_source_names)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(proposal)})
                if "error" not in proposal:
                    pending_action = {
                        "name": name,
                        "arguments": arguments,
                        "proposal": proposal,
                        "summary": _describe_proposal(name, arguments, proposal),
                    }
            else:
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps({"error": f"Unknown tool '{name}'"})}
                )

        if pending_action is not None:
            return ChatTurnResult(messages=messages, pending_action=pending_action)

    return ChatTurnResult(messages=messages, pending_action=None)


def execute_confirmed_action(
    pending_action: dict[str, Any], root: Path, client: Any, model: str, role_override: str | None = None
) -> str:
    """Perform a previously-proposed action after the user has confirmed it in the UI.

    ``role_override`` lets the user change the predictive/predicted classification shown in
    the confirmation card before it's persisted -- the model's proposed role is a first guess,
    not the final word.
    """
    name = pending_action["name"]
    proposal = pending_action["proposal"]

    if name == "propose_add_source":
        source = SourceSpec(
            name=proposal["name"],
            pathogen=proposal["pathogen"],
            description=proposal["description"],
            role=role_override or proposal["role"],
            url=proposal.get("url"),
            catalog_slug=proposal.get("catalog_slug"),
            google_trends_term=proposal.get("google_trends_term"),
            google_trends_geo=proposal.get("google_trends_geo", "GB"),
            google_trends_timeframe=proposal.get("google_trends_timeframe", "today 5-y"),
            google_trends_local_file=proposal.get("google_trends_local_file"),
        )
        add_custom_source(source, root)
        return f"Added source '{source.name}' (role: {source.role})."

    if name == "propose_run_ingestion":
        source_name = proposal["source_name"]
        source = next((candidate for candidate in list_sources(root) if candidate.name == source_name), None)
        if source is None:
            return f"Could not run ingestion: source '{source_name}' no longer exists."
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        manifest = agent_ingest.run_source_ingestion(source, root, client, model, run_id)
        return f"Ingested '{source_name}': {manifest['rows']} rows, validation_status={manifest['validation_status']}."

    return f"Unknown action '{name}'."

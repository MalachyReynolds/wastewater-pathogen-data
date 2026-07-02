"""LLM-assisted steps in the data agent pipeline.

Every function here takes the client as a parameter (so tests can inject a
stub) and never lets an LLM failure -- a bad response, a timeout, an API
error -- crash the pipeline. Each falls back to a plain heuristic when the
call fails or the response can't be parsed.
"""
from __future__ import annotations

import json
import re
from typing import Any

GEOGRAPHY_CANDIDATES = ("location", "country", "geography", "region", "area", "nation")
DATE_CANDIDATES = ("date", "period", "week", "epi_week_start", "target_date", "day")


def _strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    return match.group(1) if match else text


def _call_llm_json(client: Any, model: str, system_prompt: str, user_prompt: str) -> dict | None:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content
        return json.loads(_strip_code_fence(content))
    except Exception:
        return None


def _call_llm_text(client: Any, model: str, system_prompt: str, user_prompt: str) -> str | None:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content
        return content.strip() if content else None
    except Exception:
        return None


def _heuristic_column_mapping(columns: list[str]) -> dict[str, Any]:
    lowered = {column.lower(): column for column in columns}
    date_column = next((lowered[c] for c in DATE_CANDIDATES if c in lowered), None)
    geography_column = next((lowered[c] for c in GEOGRAPHY_CANDIDATES if c in lowered), None)
    excluded = {date_column, geography_column}
    signal_columns = [column for column in columns if column not in excluded]
    return {"date_column": date_column, "geography_column": geography_column, "signal_columns": signal_columns}


def infer_column_mapping(
    client: Any, model: str, columns: list[str], sample_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Identify the date column, an optional geography column, and signal columns.

    Falls back to name/position heuristics if the LLM call fails or returns
    something that isn't the expected JSON shape.
    """
    fallback = _heuristic_column_mapping(columns)

    system_prompt = (
        "You map raw tabular columns to a respiratory-surveillance schema. "
        "Reply with JSON only: "
        '{"date_column": <column name or null>, "geography_column": <column name or null>, '
        '"signal_columns": [<column names to treat as numeric signals>]}. '
        "Only use column names that appear in the input."
    )
    user_prompt = json.dumps({"columns": columns, "sample_rows": sample_rows[:5]})

    result = _call_llm_json(client, model, system_prompt, user_prompt)
    if not isinstance(result, dict):
        return fallback

    valid_columns = set(columns)
    date_column = result.get("date_column")
    geography_column = result.get("geography_column")
    signal_columns = result.get("signal_columns")

    if date_column not in valid_columns:
        date_column = fallback["date_column"]
    if geography_column not in valid_columns:
        geography_column = fallback["geography_column"]
    if not isinstance(signal_columns, list) or not signal_columns or not set(signal_columns) <= valid_columns:
        signal_columns = fallback["signal_columns"]

    return {"date_column": date_column, "geography_column": geography_column, "signal_columns": signal_columns}


def summarize_manifest(client: Any, model: str, feature_set: str, stats: dict[str, Any]) -> str:
    """Produce a short human-readable summary of an ingestion run for the manifest."""
    fallback = (
        f"{feature_set}: {stats.get('rows', 'unknown')} rows across "
        f"{stats.get('signal_count', 'unknown')} signals, "
        f"{stats.get('date_min', 'unknown')} to {stats.get('date_max', 'unknown')}."
    )

    system_prompt = (
        "You write a one-to-two sentence, factual summary of a respiratory-surveillance "
        "data ingestion run for a data catalogue manifest. Do not speculate beyond the given stats."
    )
    user_prompt = json.dumps({"feature_set": feature_set, "stats": stats})

    summary = _call_llm_text(client, model, system_prompt, user_prompt)
    return summary if summary else fallback


def _heuristic_anomaly_flags(stats: dict[str, Any]) -> dict[str, Any]:
    missing_fraction = stats.get("missing_value_fraction", 0.0) or 0.0
    if missing_fraction > 0.5:
        return {"validation_status": "failed", "notes": f"{missing_fraction:.0%} of values are missing."}
    if missing_fraction > 0.1:
        return {"validation_status": "warning", "notes": f"{missing_fraction:.0%} of values are missing."}
    return {"validation_status": "passed", "notes": "No heuristic anomalies detected."}


def flag_anomalies(client: Any, model: str, stats: dict[str, Any]) -> dict[str, Any]:
    """Return a validation_status ('passed'/'warning'/'failed') and explanatory notes."""
    fallback = _heuristic_anomaly_flags(stats)

    system_prompt = (
        "You review summary statistics from a data ingestion run and flag data-quality anomalies. "
        'Reply with JSON only: {"validation_status": "passed"|"warning"|"failed", "notes": <short explanation>}.'
    )
    user_prompt = json.dumps(stats)

    result = _call_llm_json(client, model, system_prompt, user_prompt)
    if not isinstance(result, dict) or result.get("validation_status") not in {"passed", "warning", "failed"}:
        return fallback

    return {"validation_status": result["validation_status"], "notes": str(result.get("notes", ""))}

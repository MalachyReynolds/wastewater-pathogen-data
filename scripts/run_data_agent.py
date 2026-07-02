#!/usr/bin/env python3
"""Run the autonomous respiratory data agent.

Fetches each source known to ``wastewater.agent.sources.list_sources`` (built-in
``PLACEHOLDER_SOURCES`` plus any added via the dashboard chat), uses an LLM (via
Nebius Token Factory) to help map columns, flag anomalies, and write a manifest
summary, then writes Parquet + manifest artifacts under
``data/normalized/`` and ``data_registry/`` -- the layout the dashboard's
Agent Data page already expects.

Requires the ``NEBIUS_API_KEY`` and ``NEBIUS_MODEL`` environment variables:

    NEBIUS_API_KEY=... NEBIUS_MODEL=... python scripts/run_data_agent.py

Run from the repository root:
    python scripts/run_data_agent.py

No sources are configured out of the box -- add a ``SourceSpec`` to
``src/wastewater/agent/sources.py`` first.

If a source's URL isn't reachable from wherever this runs (e.g. a restricted
network), download it manually elsewhere and point the agent at the local
copy instead of fetching it -- this still runs the same LLM steps and writes
the same artifacts, it just skips the network fetch:

    python scripts/run_data_agent.py --source <source_name> --local-file /path/to/file.csv
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wastewater.agent.config import load_config  # noqa: E402
from wastewater.agent.ingest import run_source_ingestion  # noqa: E402
from wastewater.agent.nebius_client import forge_nebius_client  # noqa: E402
from wastewater.agent.sources import list_sources  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=None, help="Only ingest the named source (default: all placeholder sources)")
    parser.add_argument(
        "--local-file",
        type=Path,
        default=None,
        help="Read this local CSV instead of fetching --source's URL over the network",
    )
    args = parser.parse_args()

    if args.local_file and not args.source:
        parser.error("--local-file requires --source to say which source it belongs to")

    available_sources = list_sources(ROOT)
    if not available_sources:
        print(
            "No sources are configured yet. Add a SourceSpec to "
            "src/wastewater/agent/sources.py, or add one from the dashboard chat, "
            "then re-run this script.",
            file=sys.stderr,
        )
        return 1

    sources = available_sources
    if args.source:
        sources = [source for source in available_sources if source.name == args.source]
        if not sources:
            known = ", ".join(source.name for source in available_sources)
            parser.error(f"Unknown source '{args.source}'. Known sources: {known}")

    try:
        config = load_config()
    except RuntimeError as exc:
        print(f"Cannot run the data agent: {exc}", file=sys.stderr)
        return 1

    client = forge_nebius_client(config)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    failures: list[str] = []
    for source in sources:
        try:
            print(f"Ingesting {source.name} ...", flush=True)
            raw_frame = pd.read_csv(args.local_file) if args.local_file else None
            manifest = run_source_ingestion(source, ROOT, client, config.model, run_id, raw_frame=raw_frame)
            print(
                f"Done {source.name}: {manifest['rows']} rows, "
                f"validation_status={manifest['validation_status']}",
                flush=True,
            )
        except Exception as exc:
            failures.append(source.name)
            print(f"FAILED {source.name}: {exc!r}", file=sys.stderr, flush=True)

    print(f"Ingested {len(sources) - len(failures)}/{len(sources)} sources.")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

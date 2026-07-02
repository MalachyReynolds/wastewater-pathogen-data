"""Trigger the repository's data downloaders from the dashboard.

``download_owid``/``download_weather`` are plain, fast, no-key functions and are
called synchronously. ``download_all.py``/``download_clinical_data.py`` only
expose a ``main() -> int`` entry point and can be slow (a ~50MB file, a
180-second per-file timeout, and scraping several NHS England pages), so they
are run as a background subprocess instead, matching the pattern
``scripts/run_respiratory_ml_pipeline.py`` already uses to invoke the external
downloader.
"""
from __future__ import annotations

import importlib.util
import os
import select
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class DownloadJob:
    key: str
    label: str
    warning: str
    script: str
    output_files: tuple[str, ...]


DOWNLOAD_JOBS: dict[str, DownloadJob] = {
    "wastewater": DownloadJob(
        key="wastewater",
        label="Refresh wastewater raw sources",
        warning="Downloads every source in sources.csv, including a ~50MB file; can take several minutes.",
        script="download_all.py",
        output_files=("manifest.json", "download_failures.json"),
    ),
    "clinical": DownloadJob(
        key="clinical",
        label="Refresh clinical NHS data",
        warning="Scrapes multiple NHS England pages for downloadable files; can take a few minutes.",
        script="download_clinical_data.py",
        output_files=("data/clinical/clinical_download_manifest.json",),
    ),
    "data_agent": DownloadJob(
        key="data_agent",
        label="Run the autonomous data agent",
        warning=(
            "Fetches external sources and calls the Nebius Token Factory API to help normalise them. "
            "Requires NEBIUS_API_KEY and NEBIUS_MODEL to be set in the server's environment."
        ),
        script="run_data_agent.py",
        output_files=(),
    ),
}


@dataclass
class BackgroundJob:
    process: subprocess.Popen
    log_lines: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    status: str = "running"
    return_code: int | None = None
    pending_output: str = ""


def _load_external_downloader_module(root: Path):
    script_path = Path(root) / "scripts" / "download_external_respiratory_sources.py"
    spec = importlib.util.spec_from_file_location("download_external_respiratory_sources", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def download_external_sources(root: Path) -> dict[str, list[dict[str, str]]]:
    """Download OWID COVID data and Open-Meteo weather, synchronously."""
    module = _load_external_downloader_module(root)
    owid = module.download_owid()
    weather = module.download_weather()
    return {"owid_covid": [owid], "open_meteo_weather": weather}


def start_background_download(job_key: str, root: Path) -> BackgroundJob:
    job = DOWNLOAD_JOBS[job_key]
    script_path = Path(root) / "scripts" / job.script
    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return BackgroundJob(process=process)


def poll_background_job(job: BackgroundJob) -> BackgroundJob:
    """Drain whatever output is already buffered, without blocking, and update status.

    Reads raw bytes off the pipe's file descriptor via ``select``/``os.read``
    instead of ``stdout.readline()``, which would block until the child writes
    another line -- the child (e.g. ``download_all.py``) only prints once per
    file, so a blocking readline could stall a UI status check for minutes.
    """
    stdout = job.process.stdout
    while True:
        ready, _, _ = select.select([stdout], [], [], 0)
        if not ready:
            break
        chunk = os.read(stdout.fileno(), 4096).decode("utf-8", errors="replace")
        if not chunk:
            break
        job.pending_output += chunk
        while "\n" in job.pending_output:
            line, job.pending_output = job.pending_output.split("\n", 1)
            job.log_lines.append(line)

    return_code = job.process.poll()
    if return_code is None:
        job.status = "running"
    else:
        if job.pending_output:
            job.log_lines.append(job.pending_output)
            job.pending_output = ""
        job.return_code = return_code
        job.status = "done" if return_code == 0 else "failed"
    return job

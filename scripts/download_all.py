#!/usr/bin/env python3
"""Download respiratory-pathogen wastewater viral-load data sources.

Run from the repository root:
    python scripts/download_all.py

The script reads sources.csv, downloads each URL into data/raw/, and writes
manifest.json with status, size, and SHA-256 checksums.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "sources.csv"
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "manifest.json"
FAILURES = ROOT / "download_failures.json"
USER_AGENT = "wastewater-pathogen-data/0.1 (+https://github.com/MalachyReynolds/wastewater-pathogen-data)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, out: Path, timeout: int = 180) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(out)


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    with SOURCES.open(newline="", encoding="utf-8") as f:
        rows: List[Dict[str, str]] = list(csv.DictReader(f))

    manifest = []
    failures = []
    for row in rows:
        filename = row["filename"]
        out = RAW / filename
        item = dict(row)
        item["downloaded_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            print(f"Downloading {filename} ...", flush=True)
            download(row["url"], out)
            item["status"] = "ok"
            item["size_bytes"] = out.stat().st_size
            item["sha256"] = sha256_file(out)
            item["error"] = ""
        except Exception as exc:
            item["status"] = "failed"
            item["size_bytes"] = ""
            item["sha256"] = ""
            item["error"] = repr(exc)
            failures.append(item)
            try:
                out.unlink()
            except FileNotFoundError:
                pass
            print(f"FAILED {filename}: {exc!r}", file=sys.stderr, flush=True)
        manifest.append(item)

    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    FAILURES.write_text(json.dumps(failures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ok = sum(1 for x in manifest if x["status"] == "ok")
    print(f"Done: {ok}/{len(manifest)} downloaded; manifest.json written.")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

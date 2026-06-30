#!/usr/bin/env python3
"""Download respiratory-pathogen wastewater viral-load data sources.

Run from the repository root:
    python scripts/download_all.py

The script reads sources.csv, downloads each URL into data/raw/, and writes
manifest.json with status, size, SHA-256 checksums, and any extracted ZIP
contents. ZIP sources are preserved and extracted into data/raw/<zip-stem>/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
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


def relative_to_root(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


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


def safe_zip_member_path(member_name: str) -> PurePosixPath:
    """Return a normalised safe ZIP member path, rejecting traversal."""
    candidate = PurePosixPath(member_name.replace("\\", "/"))
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"unsafe ZIP member path: {member_name!r}")
    return candidate


def extract_zip(zip_path: Path, extract_dir: Path) -> List[Dict[str, object]]:
    """Extract zip_path into extract_dir and return metadata for extracted files."""
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    extracted: List[Dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member_path = safe_zip_member_path(info.filename)
            target = extract_dir / member_path
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)

            extracted.append(
                {
                    "path": relative_to_root(target),
                    "size_bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                }
            )
    return extracted


def should_extract(row: Dict[str, str], out: Path) -> bool:
    return out.suffix.lower() == ".zip" or row.get("format", "").lower().endswith("_zip")


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
            item["path"] = relative_to_root(out)
            item["size_bytes"] = out.stat().st_size
            item["sha256"] = sha256_file(out)
            item["error"] = ""

            if should_extract(row, out):
                extract_dir = RAW / out.stem
                print(f"Extracting {filename} -> {relative_to_root(extract_dir)}/ ...", flush=True)
                item["extracted_to"] = relative_to_root(extract_dir)
                item["extracted_files"] = extract_zip(out, extract_dir)
            else:
                item["extracted_to"] = ""
                item["extracted_files"] = []
        except Exception as exc:
            item["status"] = "failed"
            item["path"] = relative_to_root(out)
            item["size_bytes"] = ""
            item["sha256"] = ""
            item["extracted_to"] = ""
            item["extracted_files"] = []
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

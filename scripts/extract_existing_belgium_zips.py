#!/usr/bin/env python3
"""Extract already-downloaded Belgium ZIP files in data/raw/.

This script does not download anything. It expects the Belgium ZIP files to
already exist in the repository and extracts them into sibling folders:

- data/raw/belgium_sars_cov_2/
- data/raw/belgium_influenza/
- data/raw/belgium_rsv/

It also writes belgium_extraction_manifest.json with extracted file metadata.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT_MANIFEST = ROOT / "belgium_extraction_manifest.json"

BELGIUM_ZIPS = [
    RAW / "belgium_sars_cov_2.zip",
    RAW / "belgium_influenza.zip",
    RAW / "belgium_rsv.zip",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def safe_zip_member_path(member_name: str) -> PurePosixPath:
    candidate = PurePosixPath(member_name.replace("\\", "/"))
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"unsafe ZIP member path: {member_name!r}")
    return candidate


def extract_zip(zip_path: Path) -> Dict[str, object]:
    if not zip_path.exists():
        return {
            "zip_path": rel(zip_path),
            "status": "missing",
            "extracted_to": "",
            "extracted_files": [],
            "error": f"{zip_path} does not exist",
        }

    extract_dir = zip_path.with_suffix("")
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
                    "path": rel(target),
                    "size_bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                }
            )

    return {
        "zip_path": rel(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "status": "ok",
        "extracted_to": rel(extract_dir),
        "extracted_files": extracted,
        "error": "",
    }


def main() -> int:
    results = []
    had_failure = False
    for zip_path in BELGIUM_ZIPS:
        print(f"Extracting {rel(zip_path)} ...", flush=True)
        try:
            result = extract_zip(zip_path)
        except Exception as exc:
            result = {
                "zip_path": rel(zip_path),
                "status": "failed",
                "extracted_to": "",
                "extracted_files": [],
                "error": repr(exc),
            }
        if result["status"] != "ok":
            had_failure = True
            print(f"FAILED {rel(zip_path)}: {result['error']}", file=sys.stderr, flush=True)
        results.append(result)

    OUT_MANIFEST.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ok = sum(1 for item in results if item["status"] == "ok")
    print(f"Extracted {ok}/{len(results)} Belgium ZIP files.")
    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())

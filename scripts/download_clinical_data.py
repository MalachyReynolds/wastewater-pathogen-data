#!/usr/bin/env python3
"""Download NHS England clinical activity data used for regression notebooks.

The downloader reads clinical_sources.csv, scrapes each NHS England statistics
page for relevant CSV links, downloads those files into data/clinical/raw/, and
writes data/clinical/clinical_download_manifest.json.

It is intentionally conservative: for IUC/NHS111 pages it keeps CSV links with
IUCADC in the URL; for A&E pages it keeps Monthly A&E CSV files and excludes
ECDS, commentary, mapping, and supplementary files.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "clinical_sources.csv"
RAW = ROOT / "data" / "clinical" / "raw"
MANIFEST = ROOT / "data" / "clinical" / "clinical_download_manifest.json"
USER_AGENT = "wastewater-pathogen-data/0.1 (+https://github.com/MalachyReynolds/wastewater-pathogen-data)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_text(url: str, timeout: int = 120) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def download(url: str, out: Path, timeout: int = 180) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as response, tmp.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(out)


def extract_links(page_url: str, page_html: str) -> List[str]:
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', page_html, flags=re.IGNORECASE)
    links = []
    for href in hrefs:
        href = html.unescape(href)
        url = urljoin(page_url, href)
        if ".csv" in url.lower():
            links.append(url)
    return sorted(set(links))


def filename_from_url(url: str) -> str:
    path = unquote(urlparse(url).path)
    name = Path(path).name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def keep_link(domain: str, url: str) -> bool:
    lower = unquote(url).lower()
    if not lower.endswith(".csv"):
        return False

    if domain == "nhs111_iuc":
        return "iucadc" in lower and "raw" in lower

    if domain == "ae_emergency_admissions":
        if "monthly" not in lower or "ae" not in lower:
            return False
        excluded = ["ecds", "supplementary", "commentary", "growth", "mapping", "quarter"]
        return not any(term in lower for term in excluded)

    return True


def iter_sources() -> Iterable[Dict[str, str]]:
    with SOURCES.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source_id = row.get("source_id", "").strip()
            if not source_id or source_id.startswith("#"):
                continue
            yield row


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    manifest = []
    failures = []
    for source in iter_sources():
        print(f"Scraping {source['source_id']} ...", flush=True)
        try:
            page = fetch_text(source["page_url"])
            links = [url for url in extract_links(source["page_url"], page) if keep_link(source["domain"], url)]
            if not links:
                raise RuntimeError("No matching CSV links found")
        except Exception as exc:
            failures.append({**source, "stage": "scrape", "error": repr(exc)})
            print(f"FAILED scrape {source['source_id']}: {exc!r}", file=sys.stderr, flush=True)
            continue

        for url in links:
            out = RAW / source["domain"] / source["year"] / filename_from_url(url)
            record = {
                **source,
                "url": url,
                "path": out.relative_to(ROOT).as_posix(),
                "downloaded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            try:
                print(f"Downloading {out.relative_to(ROOT)} ...", flush=True)
                download(url, out)
                record["status"] = "ok"
                record["size_bytes"] = out.stat().st_size
                record["sha256"] = sha256_file(out)
                record["error"] = ""
            except Exception as exc:
                record["status"] = "failed"
                record["size_bytes"] = ""
                record["sha256"] = ""
                record["error"] = repr(exc)
                failures.append(record)
                try:
                    out.unlink()
                except FileNotFoundError:
                    pass
                print(f"FAILED download {url}: {exc!r}", file=sys.stderr, flush=True)
            manifest.append(record)

    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ok = sum(1 for item in manifest if item.get("status") == "ok")
    print(f"Done: {ok}/{len(manifest)} files downloaded; manifest written to {MANIFEST.relative_to(ROOT)}")
    if failures:
        print(f"Failures: {len(failures)}", file=sys.stderr)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

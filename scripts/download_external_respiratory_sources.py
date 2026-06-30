#!/usr/bin/env python3
"""Download additional public respiratory modelling sources.

This script intentionally focuses on no-key sources that can be fetched
reproducibly from a clean checkout:

- OWID COVID-19 global dataset
- Open-Meteo historical weather for UK nation / England-region centroids

Other sources listed in ``predictive_sources.csv`` and ``predicted_sources.csv``
may require bespoke workbook parsers, geography mapping, or manual API choices,
so they are catalogued but not downloaded here yet.
"""
from __future__ import annotations

import csv
import json
import sys
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external"
OUT.mkdir(parents=True, exist_ok=True)

OWID_COVID_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

UK_WEATHER_POINTS = [
    {"geography": "England", "latitude": 52.3555, "longitude": -1.1743},
    {"geography": "Scotland", "latitude": 56.4907, "longitude": -4.2026},
    {"geography": "Wales", "latitude": 52.1307, "longitude": -3.7837},
    {"geography": "Northern_Ireland", "latitude": 54.7877, "longitude": -6.4923},
    {"geography": "North_East_England", "latitude": 54.9783, "longitude": -1.6178},
    {"geography": "North_West_England", "latitude": 53.4808, "longitude": -2.2426},
    {"geography": "Yorkshire_and_Humber", "latitude": 53.8008, "longitude": -1.5491},
    {"geography": "East_Midlands", "latitude": 52.9548, "longitude": -1.1581},
    {"geography": "West_Midlands", "latitude": 52.4862, "longitude": -1.8904},
    {"geography": "East_of_England", "latitude": 52.2053, "longitude": 0.1218},
    {"geography": "London", "latitude": 51.5072, "longitude": -0.1276},
    {"geography": "South_East_England", "latitude": 51.4545, "longitude": -0.9781},
    {"geography": "South_West_England", "latitude": 51.4545, "longitude": -2.5879},
]


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "wastewater-pathogen-data/0.1"})
    with urllib.request.urlopen(req, timeout=120) as response:
        path.write_bytes(response.read())


def download_owid() -> dict[str, str]:
    path = OUT / "owid_covid_data.csv"
    download(OWID_COVID_URL, path)
    return {"source_id": "owid_covid", "path": path.relative_to(ROOT).as_posix(), "url": OWID_COVID_URL}


def open_meteo_url(latitude: float, longitude: float, start_date: str, end_date: str) -> str:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(
            [
                "temperature_2m_mean",
                "temperature_2m_min",
                "temperature_2m_max",
                "precipitation_sum",
                "rain_sum",
                "relative_humidity_2m_mean",
                "wind_speed_10m_mean",
                "surface_pressure_mean",
            ]
        ),
        "timezone": "Europe/London",
        "format": "csv",
    }
    return "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)


def download_weather(start_date: str = "2020-01-01", end_date: str | None = None) -> list[dict[str, str]]:
    if end_date is None:
        end_date = (date.today() - timedelta(days=3)).isoformat()
    rows: list[dict[str, str]] = []
    for point in UK_WEATHER_POINTS:
        url = open_meteo_url(point["latitude"], point["longitude"], start_date, end_date)
        path = OUT / "weather" / f"open_meteo_{point['geography']}.csv"
        download(url, path)
        rows.append(
            {
                "source_id": f"open_meteo_{point['geography']}",
                "path": path.relative_to(ROOT).as_posix(),
                "url": url,
                "geography": point["geography"],
            }
        )
    return rows


def write_manifest(rows: list[dict[str, str]]) -> None:
    manifest_json = OUT / "external_download_manifest.json"
    manifest_csv = OUT / "external_download_manifest.csv"
    manifest_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for name, fn in [
        ("owid_covid", download_owid),
        ("open_meteo_weather", download_weather),
    ]:
        try:
            result = fn()
            if isinstance(result, list):
                rows.extend(result)
            else:
                rows.append(result)
        except Exception as exc:
            failures.append({"source_id": name, "error": repr(exc)})
    write_manifest(rows)
    (OUT / "external_download_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Downloaded {len(rows)} external files; failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

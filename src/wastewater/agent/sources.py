from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


VALID_ROLES = {"predictive", "predicted"}


@dataclass(frozen=True)
class SourceSpec:
    """One external source the agent can ingest, located by URL, OWID catalog slug, a live
    Google Trends term, or a local Google Trends export file.

    ``role`` is required and must be set explicitly by whoever adds the source -- it is not
    inferred from the data, since a heuristic guess (e.g. matching a keyword in the column
    name) misclassifies anything that doesn't happen to match.
    """

    name: str
    pathogen: str
    description: str
    role: str
    url: str | None = None
    catalog_slug: str | None = None
    google_trends_term: str | None = None
    google_trends_geo: str = "GB"
    google_trends_timeframe: str = "today 5-y"
    google_trends_local_file: str | None = None

    def __post_init__(self) -> None:
        location_fields = (self.url, self.catalog_slug, self.google_trends_term, self.google_trends_local_file)
        if sum(bool(field) for field in location_fields) != 1:
            raise ValueError(
                f"SourceSpec '{self.name}' must set exactly one of url, catalog_slug, "
                "google_trends_term, or google_trends_local_file."
            )
        if self.role not in VALID_ROLES:
            raise ValueError(f"SourceSpec '{self.name}' role must be one of {sorted(VALID_ROLES)}, got {self.role!r}.")


# No sources configured yet. Add SourceSpec entries here to point the agent
# at real sources -- nothing else needs to change, since
# ``ingest.run_source_ingestion`` only relies on this dataclass. Either
# fetch a plain URL:
#
# PLACEHOLDER_SOURCES: list[SourceSpec] = [
#     SourceSpec(
#         name="some_source",
#         url="https://example.org/data.csv",
#         pathogen="influenza",
#         role="predictive",
#         description="What this source is and where it comes from.",
#     )
# ]
#
# or fetch by OWID catalog slug or URL (see https://pypi.org/project/owid-catalog/):
#
# PLACEHOLDER_SOURCES: list[SourceSpec] = [
#     SourceSpec(
#         name="some_source",
#         catalog_slug="weekly-hospital-admissions-covid-per-million",
#         pathogen="COVID-19",
#         role="predicted",
#         description="What this source is and where it comes from.",
#     )
# ]
PLACEHOLDER_SOURCES: list[SourceSpec] = []

# Sources added at runtime (e.g. via the dashboard chat) persist here instead of being
# written into this file -- mutating live Python source from a running process is fragile
# and wouldn't be picked up without a reimport. This mirrors the existing
# data_registry/manifests/ + data_registry/latest/ pattern, so it's tracked in git the same way.
CUSTOM_SOURCES_PATH = Path("data_registry") / "custom_sources.json"


def load_custom_sources(root: Path) -> list[SourceSpec]:
    """Load sources added at runtime, if any have been.

    Entries written before ``role`` existed default to "predictive" (matching the
    heuristic they relied on at the time) rather than failing to load -- review and
    re-add any that should actually be "predicted" via the dashboard.
    """
    path = Path(root) / CUSTOM_SOURCES_PATH
    if not path.exists():
        return []
    entries = json.loads(path.read_text())
    for entry in entries:
        entry.setdefault("role", "predictive")
    return [SourceSpec(**entry) for entry in entries]


def list_sources(root: Path) -> list[SourceSpec]:
    """All sources the agent knows about: built-in plus runtime-added."""
    return [*PLACEHOLDER_SOURCES, *load_custom_sources(root)]


def add_custom_source(source: SourceSpec, root: Path) -> None:
    """Persist a new source, rejecting a name that already exists."""
    existing = list_sources(root)
    if any(existing_source.name == source.name for existing_source in existing):
        raise ValueError(f"A source named '{source.name}' already exists.")

    path = Path(root) / CUSTOM_SOURCES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    custom = load_custom_sources(root)
    custom.append(source)
    path.write_text(json.dumps([asdict(entry) for entry in custom], indent=2) + "\n", encoding="utf-8")

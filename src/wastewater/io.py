"""Input/output helpers for wastewater data analysis."""
from __future__ import annotations

import csv
import zipfile
from pathlib import Path
from typing import Dict

import pandas as pd


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root by walking up until sources.csv is found."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "sources.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find repository root containing sources.csv")


def load_sources(root: Path | None = None) -> pd.DataFrame:
    """Load the source catalogue."""
    root = root or find_repo_root()
    return pd.read_csv(root / "sources.csv")


def inventory_raw_files(root: Path | None = None) -> pd.DataFrame:
    """Return a file inventory for data/raw."""
    root = root or find_repo_root()
    raw = root / "data" / "raw"
    files = sorted(path for path in raw.rglob("*") if path.is_file()) if raw.exists() else []
    return pd.DataFrame(
        {
            "path": [path.relative_to(root).as_posix() for path in files],
            "suffix": [path.suffix.lower() for path in files],
            "size_mb": [path.stat().st_size / 1024**2 for path in files],
        }
    )


def _detect_separator(path: Path, encoding: str = "utf-8") -> str:
    """Guess CSV separator from the first few KB."""
    sample = path.read_text(encoding=encoding, errors="replace")[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        return dialect.delimiter
    except csv.Error:
        return ";" if sample.count(";") > sample.count(",") else ","


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV, TSV or JSON table with simple delimiter detection."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", **kwargs)
    if suffix == ".csv":
        sep = kwargs.pop("sep", _detect_separator(path))
        return pd.read_csv(path, sep=sep, **kwargs)
    if suffix == ".json":
        return pd.read_json(path, **kwargs)

    raise ValueError(f"Unsupported table format for {path}")


def read_zip_tables(zip_path: Path) -> Dict[str, pd.DataFrame]:
    """Read all CSV/TSV/JSON tables from a ZIP archive."""
    tables: Dict[str, pd.DataFrame] = {}
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            lower = name.lower()
            with zf.open(name) as handle:
                if lower.endswith(".tsv"):
                    tables[name] = pd.read_csv(handle, sep="\t")
                elif lower.endswith(".csv"):
                    tables[name] = pd.read_csv(handle)
                elif lower.endswith(".json"):
                    tables[name] = pd.read_json(handle)
    return tables


def inspect_source_schemas(root: Path | None = None, max_rows: int = 5) -> list[dict]:
    """Inspect shapes and column names for every downloaded source.

    This returns metadata rather than full dataframes so it is safe to display in
    a notebook without loading very large outputs into the page.
    """
    root = root or find_repo_root()
    raw = root / "data" / "raw"
    sources = load_sources(root)
    records: list[dict] = []

    for row in sources.to_dict(orient="records"):
        path = raw / row["filename"]
        base = {
            "country": row["country"],
            "pathogen_scope": row["pathogen_scope"],
            "level": row["level"],
            "source_file": row["filename"],
            "exists": path.exists(),
        }
        if not path.exists():
            records.append({**base, "member": "", "n_rows": None, "n_cols": None, "columns": []})
            continue

        try:
            if path.suffix.lower() == ".zip":
                for member, df in read_zip_tables(path).items():
                    records.append(
                        {**base, "member": member, "n_rows": len(df), "n_cols": len(df.columns), "columns": list(df.columns)}
                    )
            else:
                df = read_table(path, nrows=max_rows)
                records.append({**base, "member": "", "n_rows": None, "n_cols": len(df.columns), "columns": list(df.columns)})
        except Exception as exc:
            records.append({**base, "member": "", "n_rows": None, "n_cols": None, "columns": [], "error": repr(exc)})

    return records

# Wastewater pathogen data

Public source catalogue, reproducible downloader, and analysis notebooks for viral-load data on respiratory pathogens in wastewater, focused on Western European countries.

The repository is designed to be refreshed from source rather than maintained by hand. The authoritative sources are listed in `sources.csv`; `scripts/download_all.py` downloads them into `data/raw/` and writes `manifest.json` plus `download_failures.json`.

## Included source types

Current sources include wastewater viral-load or wastewater RNA datasets for:

- Switzerland / Liechtenstein: SARS-CoV-2, influenza, RSV
- Germany: SARS-CoV-2, influenza A/B, RSV A/B
- Belgium: SARS-CoV-2, influenza, RSV
- France: SARS-CoV-2 SUM'Eau indicators
- Netherlands: SARS-CoV-2 wastewater data
- Scotland: SARS-CoV-2 wastewater RNA

## Refresh the data

```bash
python scripts/download_all.py
```

The script continues after individual download failures and records them in `download_failures.json`.

## Analyse the data

Create an environment, install dependencies, and open JupyterLab:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
jupyter lab
```

Then open:

```text
notebooks/01_wastewater_analysis.ipynb
```

The first notebook inventories `data/raw/`, inspects source schemas, defines a canonical long-format target, and provides placeholders for country-specific cleaning adapters.

## NHS111 / admissions regression

Download NHS England clinical activity data with:

```bash
python scripts/download_clinical_data.py
```

This reads `clinical_sources.csv`, scrapes the NHS England Integrated Urgent Care and A&E statistics pages, and writes files under:

```text
data/clinical/raw/
```

Then open:

```text
notebooks/02_nhs111_gp_regression.ipynb
```

The notebook starts with a national monthly regression of admission activity on NHS111/IUC call activity and lagged call activity. If the intended outcome is a more specific GP measure, use the notebook's outcome-column selection step to swap in the relevant GP-related field.

Reusable notebook helpers live in:

```text
src/wastewater/
```

## GitHub Actions

A weekly workflow is included at `.github/workflows/update-data.yml`. It can also be run manually from the Actions tab. The workflow downloads the latest raw files into `data/raw/` and commits any changes.

The Belgium ZIP extraction workflow is included at `.github/workflows/extract-belgium-zips.yml` and can be run manually from the Actions tab.

## Notes

Some portals expose large CSV/TSV files. The largest currently known source is the Germany individual-site AMELAG file, which is roughly 50 MB. Large raw files are best fetched directly by the downloader or GitHub Actions rather than uploaded through an interactive connector.

Check upstream licences and attribution requirements before publication or redistribution.

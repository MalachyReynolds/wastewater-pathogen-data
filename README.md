# Wastewater pathogen data

Public source catalogue and reproducible downloader for viral-load data on respiratory pathogens in wastewater, focused on Western European countries.

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

## GitHub Actions

A weekly workflow is included at `.github/workflows/update-data.yml`. It can also be run manually from the Actions tab. The workflow downloads the latest raw files into `data/raw/` and commits any changes.

## Notes

Some portals expose large CSV/TSV files. The largest currently known source is the Germany individual-site AMELAG file, which is roughly 50 MB. Large raw files are best fetched directly by the downloader or GitHub Actions rather than uploaded through an interactive connector.

Check upstream licences and attribution requirements before publication or redistribution.

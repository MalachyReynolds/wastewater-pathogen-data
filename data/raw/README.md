# Raw data

Raw datasets are downloaded here by:

```bash
python scripts/download_all.py
```

The downloader writes all source files from `sources.csv` into this directory, then updates `manifest.json` and `download_failures.json` in the repository root.

ZIP sources are preserved and also extracted into sibling folders named after the ZIP stem. For example, the Belgium sources download as:

- `data/raw/belgium_sars_cov_2.zip`
- `data/raw/belgium_influenza.zip`
- `data/raw/belgium_rsv.zip`

and are extracted into:

- `data/raw/belgium_sars_cov_2/`
- `data/raw/belgium_influenza/`
- `data/raw/belgium_rsv/`

The large raw files are intentionally fetched directly from upstream sources rather than streamed through the chat connector.

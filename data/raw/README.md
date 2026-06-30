# Raw data

Raw datasets are downloaded here by:

```bash
python scripts/download_all.py
```

The downloader writes all source files from `sources.csv` into this directory, then updates `manifest.json` and `download_failures.json` in the repository root.

The large raw files are intentionally fetched directly from upstream sources rather than streamed through the chat connector.

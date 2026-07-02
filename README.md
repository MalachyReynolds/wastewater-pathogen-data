# Wastewater pathogen data

Public source catalogue, reproducible downloader, and analysis notebooks for viral-load data on respiratory pathogens in wastewater, focused on Western European countries.

The repository is designed to be refreshed from source rather than maintained by hand. The authoritative wastewater sources are listed in `sources.csv`; `scripts/download_all.py` downloads them into `data/raw/` and writes `manifest.json` plus `download_failures.json`.

## Included source types

Current sources include wastewater viral-load or wastewater RNA datasets for:

- Switzerland / Liechtenstein: SARS-CoV-2, influenza, RSV
- Germany: SARS-CoV-2, influenza A/B, RSV A/B
- Belgium: SARS-CoV-2, influenza, RSV
- France: SARS-CoV-2 SUM'Eau indicators
- Netherlands: SARS-CoV-2 wastewater data
- Scotland: SARS-CoV-2 wastewater RNA

Additional predictive and predicted source catalogues are listed in:

```text
predictive_sources.csv
predicted_sources.csv
```

## Refresh the data

```bash
python scripts/download_all.py
python scripts/download_clinical_data.py
python scripts/download_external_respiratory_sources.py
```

The wastewater downloader continues after individual download failures and records them in `download_failures.json`. The external downloader currently fetches no-key sources that can be automated immediately, including OWID COVID data and Open-Meteo historical weather for UK nation / England-region centroids.

## Streamlit dashboard

Run the interactive dashboard with:

```bash
streamlit run app.py
```

The dashboard includes pages for loading data, exploring time series, fitting models, forecasting, refreshing downloads, loading autonomous-agent artifacts, and estimating superspreader-event risk.

## Autonomous data agent

The `Agent Data` page consumes artifacts written by an autonomous ingestion agent under
`src/wastewater/agent/`. The agent fetches a source, calls an LLM (Nebius Token Factory's
OpenAI-compatible API) to help map columns, flag data-quality anomalies, and write a manifest
summary, then writes:

```text
data/normalized/<source_name>/<run_id>.parquet         # normalized long-format signal table
data_registry/manifests/<source_name>/<run_id>.json    # full manifest
data_registry/latest/<source_name>.json                # pointer to the latest manifest
```

Run it from the CLI, or click "Start data agent run" on the `Agent Data` page (which runs the
same script as a background job). Either way it requires two environment variables set in
whichever process runs it -- never entered through the browser:

```bash
NEBIUS_API_KEY=...   # your Nebius Token Factory API key
NEBIUS_MODEL=...     # a model slug available on your Nebius account
python scripts/run_data_agent.py
```

`NEBIUS_MODEL` is required with no hardcoded default, since available model slugs vary by
account. If the LLM call fails or returns something unparseable, each step falls back to a
plain heuristic (column-name/dtype matching for schema mapping, a missing-value threshold for
anomaly flagging, a templated sentence for the summary) so a flaky or unreachable API never
breaks the pipeline -- it just produces a less-informed manifest.

If a source's URL isn't reachable from wherever the script runs (some networks don't resolve
every public data host), download it manually and point the agent at the local file instead --
this still runs the same LLM steps and writes the same artifacts, it just skips the fetch:

```bash
python scripts/run_data_agent.py --source <source_name> --local-file /path/to/file.csv
```

No sources are configured out of the box. Add one by adding a `SourceSpec` entry to
`PLACEHOLDER_SOURCES` in `src/wastewater/agent/sources.py` -- nothing else needs to change. A
source is located by exactly one of two fields:

- `url`: a plain CSV endpoint, fetched over HTTP. The LLM infers which columns are the date,
  geography, and signal columns, since the schema isn't known in advance.
- `catalog_slug`: a dataset slug from [OWID's data catalog](https://pypi.org/project/owid-catalog/)
  (e.g. `weekly-hospital-admissions-covid-per-million`), fetched via `owid.catalog.fetch`. The
  column mapping is derived directly from the catalog table's own index, so no LLM call is
  needed for that step -- the LLM is still used for the anomaly-flagging and summary steps.

## Superspreader event risk tool

The `Superspreader Risk` Streamlit page estimates whether a candidate event could disproportionately amplify transmission for RSV, influenza, COVID-19, or another respiratory pathogen. It uses an interpretable MVP model under:

```text
src/wastewater/superspreading/
```

For each candidate event, the tool reports:

```text
Transmission Amplification Factor (TAF)
Expected event transmission
P(SSE), the probability of exceeding a homogeneous superspreading threshold
90% simulated interval for secondary infections
Risk band
Optional regional contribution index
```

The main mathematical measure is:

```text
TAF = E[secondary infections | event conditions] / E[secondary infections | baseline conditions]
```

The simulator draws infectious attendees from a binomial model and secondary infections from a negative-binomial offspring distribution, using local prevalence, R_t, event size, event conditions, and a dispersion parameter `k`. The coefficients are transparent defaults for scenario comparison; they should be calibrated with outbreak investigations or inferred event impacts before operational use.

## Run an end-to-end model

For a command-line run that downloads accessible external data, builds the panel, trains models, and writes outputs, run:

```bash
python scripts/run_respiratory_ml_pipeline.py
```

This writes diagnostics and model outputs to:

```text
data/processed/respiratory_ml_canonical_series.csv
data/processed/respiratory_ml_series_summary.csv
data/processed/respiratory_ml_family_counts.csv
data/processed/respiratory_incidence_ml_model_results.csv
data/processed/respiratory_incidence_ml_model_predictions.csv
```

Use this first when checking whether the repository can actually access enough data to fit at least one model.

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

## Respiratory incidence ML panel

To train multiple machine-learning models on the larger predictive/predicted respiratory-virus panel, open:

```text
notebooks/06_respiratory_incidence_ml_panel.ipynb
```

This notebook gathers currently available canonical series, builds lagged features, and compares OLS, ridge, elastic net, random forest, and histogram gradient boosting under a chronological train-test split. Outputs are written to:

```text
data/processed/respiratory_incidence_ml_model_results.csv
data/processed/respiratory_incidence_ml_model_predictions.csv
```

The supporting code lives in:

```text
src/wastewater/ml_panel.py
src/wastewater/external_series.py
```

## Leakage-safe predictive vs predicted matrix

To compare every predictive/predicted pair without future leakage, open:

```text
notebooks/05_predictive_vs_predicted_train_test_matrix.ipynb
```

This workflow uses expanding-window forecasts. For each prediction period, the model is trained only on earlier periods, uses positive predictor lags only, and scores only out-of-sample predictions. It reports best and worst held-out errors across all pairs and writes spike-risk scores for hospital-admission targets.

Predictive series currently include:

- Google Trends one-year files in `Google_trends_v2/1y_data/time_series_GB*`
- UKHSA charts classified as NHS-call series
- raw wastewater files listed in `sources.csv` and stored under `data/raw/`
- processed wastewater long-format data, if `data/processed/wastewater_long.{parquet,csv}` also exists

Predicted series currently include:

- UKHSA charts classified as GP/admission series

It writes results to:

```text
data/processed/leakage_safe_pairwise_forecast_results.csv
data/processed/leakage_safe_pairwise_forecast_predictions.csv
data/processed/leakage_safe_hospital_spike_scores.csv
```

The supporting code lives in:

```text
src/wastewater/leakage_safe_matrix.py
src/wastewater/regression_matrix.py
```

## UKHSA NHS calls / GP admissions regression

For the existing UKHSA dashboard export files in the repo, open:

```text
notebooks/03_ukhsa_nhs111_gp_regression.ipynb
```

This notebook scans the local checkout for files beginning with `ukhsa-chart`, infers date and value columns, classifies files into NHS-call predictors and GP/admission outcomes, and fits lagged OLS regressions. The supporting code lives in:

```text
src/wastewater/ukhsa.py
```

## Google Trends 1-year / GP admissions regression

For the one-year Google Trends search-term files under:

```text
Google_trends_v2/1y_data/
```

open:

```text
notebooks/04_search_terms_gp_admissions_regression.ipynb
```

This notebook scans `Google_trends_v2/1y_data` for `time_series_GB*` files, uses the second source column from each file as a Google Trends predictor, and regresses GP admissions on contemporaneous and lagged Google Trends predictors. It includes a chronological train-test split to check held-out predictive performance, reporting RMSE, MAE, R², correlation, and improvement over a training-mean baseline. The supporting code lives in:

```text
src/wastewater/search_terms.py
```

## NHS England clinical downloader workflow

A separate NHS England downloader workflow is also available. Download NHS England clinical activity data with:

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

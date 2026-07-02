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

The dashboard includes pages for loading the canonical series panel, exploring signals, fitting models, forecasting, downloading outputs, and loading autonomous-agent artifacts.

The **Agent Data** page expects the lightweight agent storage layout:

```text
data_registry/latest/*.json          # latest pointers
data_registry/manifests/**/*.json    # full artifact manifests
data/features/**/*.parquet           # model-ready feature tables
data/normalized/**/*.parquet         # normalized long-form signal tables
```

A latest pointer should either be a full manifest or point to one:

```json
{
  "feature_set": "england_rsv_admissions_2w",
  "latest_run_id": "20260702T120000Z",
  "manifest_path": "data_registry/manifests/england_rsv_admissions_2w/20260702T120000Z.json"
}
```

The referenced manifest should include a `path` to a Parquet feature table. Once loaded, the Agent Data page converts the table into the same canonical long-format series panel used by the existing Explore, Model, and Forecast pages.

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

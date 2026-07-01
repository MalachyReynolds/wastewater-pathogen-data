#!/usr/bin/env python3
"""Run an end-to-end respiratory incidence ML pipeline.

This script is intended to be runnable from a fresh checkout. By default it:

1. downloads no-key external sources that are immediately automatable,
2. loads all locally available predictive and predicted series,
3. builds lagged supervised panels,
4. trains several models under a chronological train-test split,
5. writes model results and predictions to ``data/processed``.

It exits non-zero with clear diagnostics if there is not enough accessible data
to fit at least one model.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wastewater.external_series import build_all_available_series  # noqa: E402
from wastewater.ml_panel import PanelBuildConfig, evaluate_all_targets  # noqa: E402
from wastewater.regression_matrix import summarise_available_series  # noqa: E402


def parse_lags(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def run_external_downloader() -> int:
    script = ROOT / "scripts" / "download_external_respiratory_sources.py"
    print(f"Running {script.relative_to(ROOT)}")
    completed = subprocess.run([sys.executable, str(script)], cwd=ROOT, check=False)
    return int(completed.returncode)


def write_diagnostics(series: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    series.to_csv(out_dir / "respiratory_ml_canonical_series.csv", index=False)
    summary = summarise_available_series(series)
    summary.to_csv(out_dir / "respiratory_ml_series_summary.csv", index=False)
    family_counts = (
        series.groupby(["role", "dataset_family"], dropna=False)["series_id"]
        .nunique()
        .reset_index(name="n_series")
        .sort_values(["role", "dataset_family"])
        if not series.empty
        else pd.DataFrame(columns=["role", "dataset_family", "n_series"])
    )
    family_counts.to_csv(out_dir / "respiratory_ml_family_counts.csv", index=False)
    print("\nAvailable series families:")
    if family_counts.empty:
        print("  none")
    else:
        print(family_counts.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-download-external", action="store_true", help="Do not run the external no-key downloader first")
    parser.add_argument("--freq", default="W", choices=["W", "M"], help="Aggregation frequency")
    parser.add_argument("--lags", default="1,2,3,4,5,6,7,8", help="Comma-separated predictive lags")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--min-test-size", type=int, default=4)
    parser.add_argument("--min-non-missing-fraction", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_download_external:
        code = run_external_downloader()
        if code != 0:
            print("External downloader reported failures; continuing with whatever data was downloaded.")

    series = build_all_available_series(ROOT, include_external=True)
    write_diagnostics(series, out_dir)

    predictive_count = int(series.loc[series["role"] == "predictive", "series_id"].nunique()) if not series.empty else 0
    predicted_count = int(series.loc[series["role"] == "predicted", "series_id"].nunique()) if not series.empty else 0
    if predictive_count == 0 or predicted_count == 0:
        print("\nCannot fit a model because accessible data are incomplete.")
        print(f"Predictive series found: {predictive_count}")
        print(f"Predicted series found: {predicted_count}")
        print("Run the download scripts or check that local UKHSA/Google Trends/wastewater files exist.")
        return 2

    config = PanelBuildConfig(
        freq=args.freq,
        lags=parse_lags(args.lags),
        aggregation="mean",
        min_non_missing_fraction=args.min_non_missing_fraction,
    )
    print("\nTraining models with config:", config)
    results, predictions = evaluate_all_targets(
        series,
        config=config,
        train_fraction=args.train_fraction,
        min_test_size=args.min_test_size,
        random_state=42,
    )

    results_path = out_dir / "respiratory_incidence_ml_model_results.csv"
    predictions_path = out_dir / "respiratory_incidence_ml_model_predictions.csv"
    results.to_csv(results_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    print(f"\nSaved model results to {results_path.relative_to(ROOT)}")
    print(f"Saved model predictions to {predictions_path.relative_to(ROOT)}")

    ok = results[results["status"] == "ok"].copy() if not results.empty and "status" in results.columns else pd.DataFrame()
    if ok.empty:
        print("\nNo models fitted successfully. Top errors:")
        if not results.empty:
            print(results[[col for col in ["target_id", "model", "error"] if col in results.columns]].head(20).to_string(index=False))
        return 3

    ranked = ok.sort_values("mse_skill_vs_train_mean", ascending=False)
    cols = ["target_id", "model", "n_train", "n_test", "n_features", "rmse", "baseline_rmse", "mse_skill_vs_train_mean", "correlation", "r2"]
    print("\nTop held-out model results:")
    print(ranked[[col for col in cols if col in ranked.columns]].head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

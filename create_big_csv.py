from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "big_spine.csv"

# Set these to trim the output date range.
# Leave either one as None to keep the range open-ended.
DATE_START = None  # example: "2020-01-01"
DATE_END = None    # example: "2024-12-31"


# Edit this list to control which columns end up in the wide file.
# If a source has a series column, each unique series becomes its own output column.
DATASETS = [
	{
		"name": "uk_ari_calls",
		"output_name": "uk_ari_calls",
		"path": BASE_DIR / "ukhsa-chart-Acute-Respiratory-Infection-NHS-111-Calls-Daily.csv",
		"read_csv_kwargs": {},
		"date_column": "date",
		"series_column": "metric",
		"value_column": "metric_value",
		"series_keep": None,
		"series_rename": {},
		"filters": {
			"geography": "England",
			"sex": "all",
			"age": "all",
			"stratum": "default",
		},
	},
	{
		"name": "uk_ari_syndromic_calls",
		"output_name": "uk_ari_syndromic_calls",
		"path": BASE_DIR / "ukhsa-chart-Acute-Respiratory-Infection-Syndromic-NHS-111-calls-Daily.csv",
		"read_csv_kwargs": {},
		"date_column": "date",
		"series_column": "metric",
		"value_column": "metric_value",
		"series_keep": None,
		"series_rename": {},
		"filters": {
			"geography": "England",
			"sex": "all",
			"age": "all",
			"stratum": "default",
		},
	},
	{
		"name": "uk_influenza_like_ed",
		"output_name": "uk_influenza_like_ed",
		"path": BASE_DIR / "ukhsa-chart-Influenza-Like-Syndromic-Emergency-Department-Admissions-Daily.csv",
		"read_csv_kwargs": {},
		"date_column": "date",
		"series_column": "metric",
		"value_column": "metric_value",
		"series_keep": None,
		"series_rename": {},
		"filters": {
			"geography": "England",
			"sex": "all",
			"age": "all",
			"stratum": "default",
		},
	},
	{
		"name": "uk_lrti_gp_hours",
		"output_name": "uk_lrti_gp_hours",
		"path": BASE_DIR / "ukhsa-chart-Lower-Respiratory-Tract-Infection-GP-Hours-Daily.csv",
		"read_csv_kwargs": {},
		"date_column": "date",
		"series_column": "metric",
		"value_column": "metric_value",
		"series_keep": None,
		"series_rename": {},
		"filters": {
			"geography": "England",
			"sex": "all",
			"age": "all",
			"stratum": "default",
		},
	},
	{
		"name": "uk_urti_gp_hours",
		"output_name": "uk_urti_gp_hours",
		"path": BASE_DIR / "ukhsa-chart-Upper-Respiratory-Tract-Infection-GP-Hours-Daily.csv",
		"read_csv_kwargs": {},
		"date_column": "date",
		"series_column": "metric",
		"value_column": "metric_value",
		"series_keep": None,
		"series_rename": {},
		"filters": {
			"geography": "England",
			"sex": "all",
			"age": "all",
			"stratum": "default",
		},
	},
	{
		"name": "netherlands_rna",
		"output_name": "netherlands_rna",
		"path": BASE_DIR / "data/raw/netherlands_rivm_national.csv",
		"read_csv_kwargs": {"sep": ";"},
		"date_column": "Date_measurement",
		"series_column": None,
		"value_column": "RNA_flow_per_100000",
		"series_keep": None,
		"series_rename": {},
		"filters": {},
	},
	{
		"name": "scotland_rna",
		"output_name": "scotland_rna",
		"path": BASE_DIR / "data/raw/scotland_wastewater_covid19_national.csv",
		"read_csv_kwargs": {},
		"date_column": "SevenDayEnding",
		"date_format": "%Y%m%d",
		"series_column": None,
		"value_column": "WastewaterRNA",
		"series_keep": None,
		"series_rename": {},
		"filters": {},
	},
	{
		"name": "switzerland_rna",
		"output_name": "switzerland_rna",
		"path": BASE_DIR / "data/raw/switzerland_liechtenstein_respiratory_wastewater.csv",
		"read_csv_kwargs": {"low_memory": False},
		"date_column": "temporal",
		"series_column": "valueCategory",
		"value_column": "value",
		"series_keep": ["sars-cov-2", "influenza_a", "influenza_b", "respiratory_syncytial_virus"],
		"series_rename": {
			"sars-cov-2": "swiss_sars_cov_2",
			"influenza_a": "swiss_influenza_a",
			"influenza_b": "swiss_influenza_b",
			"respiratory_syncytial_virus": "swiss_rsv",
		},
		"filters": {},
	},
]


def load_series(dataset: dict) -> pd.DataFrame:
	frame = pd.read_csv(dataset["path"], **dataset["read_csv_kwargs"])

	for column_name, expected_value in dataset["filters"].items():
		if column_name in frame.columns:
			frame = frame.loc[frame[column_name] == expected_value].copy()

	date_parse_kwargs = {"errors": "coerce"}
	if dataset.get("date_format"):
		date_parse_kwargs["format"] = dataset["date_format"]
	frame[dataset["date_column"]] = pd.to_datetime(frame[dataset["date_column"]], **date_parse_kwargs)
	frame = frame.loc[frame[dataset["date_column"]].notna()].copy()
	frame["date"] = frame[dataset["date_column"]].dt.normalize()

	if dataset["series_column"] is None:
		series = frame[["date", dataset["value_column"]]].copy()
		series = series.rename(columns={dataset["value_column"]: dataset["output_name"]})
		return series.groupby("date", as_index=False).first()

	series = frame[["date", dataset["series_column"], dataset["value_column"]]].copy()
	series[dataset["series_column"]] = series[dataset["series_column"]].astype(str)
	if dataset.get("series_keep"):
		series = series.loc[series[dataset["series_column"]].isin(dataset["series_keep"])].copy()
		series[dataset["series_column"]] = pd.Categorical(
			series[dataset["series_column"]], categories=dataset["series_keep"], ordered=True
		)
	series["column_name"] = series[dataset["series_column"]].map(dataset["series_rename"]).fillna(
		series[dataset["series_column"]].astype(str).map(lambda series_name: f'{dataset["name"]}__{series_name}')
	)
	wide = series.pivot_table(
		index="date",
		columns="column_name",
		values=dataset["value_column"],
		aggfunc="first",
		observed=False,
	)
	wide = wide.reset_index()
	return wide


def ordered_output_columns() -> list[str]:
	columns = ["date"]

	for dataset in DATASETS:
		if dataset["series_column"] is None:
			columns.append(dataset["output_name"])
			continue

		if dataset.get("series_keep"):
			columns.extend(
				dataset["series_rename"].get(series_name, f'{dataset["name"]}__{series_name}')
				for series_name in dataset["series_keep"]
			)
			continue

		columns.append(dataset["output_name"])

	return columns


def build_spine() -> pd.DataFrame:
	frames = [load_series(dataset) for dataset in DATASETS]
	merged = frames[0]

	for frame in frames[1:]:
		merged = merged.merge(frame, on="date", how="outer")

	columns = [column_name for column_name in ordered_output_columns() if column_name in merged.columns]
	remaining_columns = [column_name for column_name in merged.columns if column_name not in columns]
	merged = merged[columns + remaining_columns]
	merged = merged.sort_values("date").reset_index(drop=True)

	if DATE_START is not None:
		merged = merged.loc[merged["date"] >= pd.to_datetime(DATE_START)]
	if DATE_END is not None:
		merged = merged.loc[merged["date"] <= pd.to_datetime(DATE_END)]

	merged = merged.reset_index(drop=True)
	return merged


def main() -> None:
	spine = build_spine()
	spine.to_csv(OUTPUT_PATH, index=False)
	print(f"Wrote {OUTPUT_PATH} with {len(spine)} rows and {len(spine.columns) - 1} data columns")


if __name__ == "__main__":
	main()




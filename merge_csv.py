import pandas as pd

# Load the two CSV files
big_spine = pd.read_csv("big_spine.csv")
seasonal_features = pd.read_csv("seasonal_features.csv")

# Ensure date columns are datetime type for proper merging
big_spine["date"] = pd.to_datetime(big_spine["date"])
seasonal_features["Date"] = pd.to_datetime(seasonal_features["Date"])

# Merge on date (left join to keep all dates from big_spine)
merged = big_spine.merge(
    seasonal_features.rename(columns={"Date": "date"}),
    on="date",
    how="left"
)

# Save the merged result
merged.to_csv("merged_data.csv", index=False)

print(f"Merged {len(big_spine)} rows from big_spine.csv with seasonal_features.csv")
print(f"Output saved to merged_data.csv with {len(merged.columns)} columns")
print(f"\nFirst few rows:")
print(merged.head())


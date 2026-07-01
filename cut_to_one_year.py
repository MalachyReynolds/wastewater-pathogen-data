import pandas as pd

df = pd.read_csv("merged_data.csv")
df["date"] = pd.to_datetime(df["date"])

# Filter between two dates
start_date = "2025-06-22"
end_date = "2026-06-18"

filtered = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

# Save the result
filtered.to_csv("filtered_data.csv", index=False)
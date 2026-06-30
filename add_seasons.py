import pandas as pd
import numpy as np

# Create a weekly date range (change the dates as needed)

dates = pd.date_range(start="2020-05-28", end="2026-06-22", freq="D")

# Create DataFrame

seasonal_df = pd.DataFrame({"Date": dates})

temp = pd.read_csv("meantemp_daily_totals.txt", sep=r"\s+", engine="python")
temp = temp.rename(columns={"Value": "Temp"})
temp["Date"] = pd.to_datetime(temp["Date"], errors="coerce")
temp["Temp"] = pd.to_numeric(temp["Temp"], errors="coerce")
temp = temp.loc[temp["Date"].notna()].copy()

seasonal_df = seasonal_df.merge(temp[["Date", "Temp"]], on="Date", how="left")

# Week of year

seasonal_df["WeekOfYear"] = seasonal_df["Date"].dt.isocalendar().week.astype(int)

# Cyclical encoding of week number

seasonal_df["SinWeek"] = np.sin(2 * np.pi * seasonal_df["WeekOfYear"] / 52)

seasonal_df["CosWeek"] = np.cos(2 * np.pi * seasonal_df["WeekOfYear"] / 52)


# Save to CSV

seasonal_df.to_csv("seasonal_features.csv", index=False)

print(seasonal_df.head())
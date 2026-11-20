from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STEP = 0.5

weather = pd.read_csv(ROOT / "data" / "raw" / "power_all.csv", parse_dates=["date"])
grid = pd.read_csv(ROOT / "data" / "processed" / "grid_cells.csv")
fires = pd.read_csv(ROOT / "data" / "processed" / "fires_tunisia_clean.csv", parse_dates=["acq_date"])

# Friendlier column names
weather = weather.rename(columns={
    "T2M": "temp_mean", "T2M_MAX": "temp_max", "T2M_MIN": "temp_min",
    "RH2M": "humidity", "WS2M": "wind", "PRECTOTCORR": "rain",
})

# ---- 1. Count fires per cell per day ----
fires["cell_id"] = (np.floor(fires["latitude"] / STEP).astype(int).astype(str) + "_"
                    + np.floor(fires["longitude"] / STEP).astype(int).astype(str))
daily = (fires.groupby(["cell_id", "acq_date"]).size()
         .rename("fires_today").reset_index()
         .rename(columns={"acq_date": "date"}))

df = weather.merge(daily, on=["cell_id", "date"], how="left")
df["fires_today"] = df["fires_today"].fillna(0)
df = df.sort_values(["cell_id", "date"]).reset_index(drop=True)   # order matters for rolling
g = df.groupby("cell_id")

def roll(col, n, how):
    """Rolling window of n days ending today, computed separately for each cell."""
    return g[col].transform(lambda s: getattr(s.rolling(n), how)())

# ---- 2. Recent fire history (today and the past) ----
df["fires_last3"] = roll("fires_today", 3, "sum")
df["fires_last7"] = roll("fires_today", 7, "sum")
df["fires_last30"] = roll("fires_today", 30, "sum")

# ---- 3. Recent weather (dryness builds up) ----
df["rain_7d"] = roll("rain", 7, "sum")
df["rain_30d"] = roll("rain", 30, "sum")
df["temp_7d"] = roll("temp_mean", 7, "mean")
df["humidity_7d"] = roll("humidity", 7, "mean")

# ---- 4. Label: is there a fire in this cell tomorrow? ----
df["next_day_fires"] = g["fires_today"].shift(-1)

# ---- 5. Calendar features ----
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
doy = df["date"].dt.day_of_year
df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

# ---- 6. Cell location ----
df = df.merge(grid[["cell_id", "lat_c", "lon_c", "frac_inside"]], on="cell_id", how="left")

# ---- 7. Drop rows without full history or without a known label ----
df = df.dropna(subset=["fires_last30", "rain_30d", "next_day_fires"]).copy()
df["fire_tomorrow"] = (df["next_day_fires"] > 0).astype(int)
df = df.drop(columns=["next_day_fires"])

out = ROOT / "data" / "processed" / "firecast_dataset.csv.gz"
df.to_csv(out, index=False)          # .gz = compressed, small enough for GitHub

# ---- Reports ----
print("Shape:", df.shape, "(expected 409182 rows)")
print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
print("Missing values:", int(df.isna().sum().sum()))
print("\nClass balance (fire_tomorrow):")
print(df["fire_tomorrow"].value_counts())
print(f"Positive rate: {df['fire_tomorrow'].mean():.2%}")
print("\nPositive rate per year:")
print(df.groupby("year")["fire_tomorrow"].mean().map("{:.2%}".format))
print("\nPositive rate per month:")
print(df.groupby("month")["fire_tomorrow"].mean().map("{:.2%}".format))
print("\nColumns:", df.columns.tolist())
print("\nSaved to:", out)

# Outlier preview: the 5 rainiest cell-days
print("\nTop 5 rainfall values:")
print(df.nlargest(5, "rain")[["cell_id", "date", "rain", "humidity"]])
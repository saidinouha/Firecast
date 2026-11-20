import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

df = pd.read_csv(PROC / "firecast_dataset.csv.gz", parse_dates=["date"])

# 1. Sort chronologically (needed later for time-ordered cross-validation)
df = df.sort_values(["date", "cell_id"]).reset_index(drop=True)

# 2. log1p transform for skewed, zero-heavy variables
log_cols = ["rain", "rain_7d", "rain_30d",
            "fires_today", "fires_last3", "fires_last7", "fires_last30"]
for c in log_cols:
    df[c + "_log"] = np.log1p(df[c])

# 3. Candidate features (order = preference when two features are redundant)
candidates = [
    "temp_max", "temp_min", "temp_mean", "temp_7d",
    "humidity", "humidity_7d", "wind",
    "rain_log", "rain_7d_log", "rain_30d_log",
    "fires_today_log", "fires_last3_log", "fires_last7_log", "fires_last30_log",
    "doy_sin", "doy_cos", "lat_c", "lon_c", "frac_inside",
]

# 4. Time-based split
train = df[df["date"] < "2022-01-01"].copy()
test = df[df["date"] >= "2022-01-01"].copy()

# 5. Correlation filter, computed on TRAIN only
corr = train[candidates].corr().abs()
kept, dropped = [], {}
for f in candidates:
    partner = next((k for k in kept if corr.loc[f, k] > 0.9), None)
    if partner is None:
        kept.append(f)
    else:
        dropped[f] = partner

# 6. Save
meta = ["cell_id", "date"]
train[meta + kept + ["fire_tomorrow"]].to_csv(PROC / "train.csv.gz", index=False)
test[meta + kept + ["fire_tomorrow"]].to_csv(PROC / "test.csv.gz", index=False)
with open(PROC / "features.json", "w") as f:
    json.dump({"features": kept, "dropped_correlated": dropped}, f, indent=2)

# ---- Reports ----
for name, d in [("TRAIN", train), ("TEST", test)]:
    pos = int(d["fire_tomorrow"].sum())
    print(f"{name}: {len(d)} rows, {d['date'].min().date()} to {d['date'].max().date()}, "
          f"fires = {pos} ({pos / len(d):.2%})")
print(f"Negative/positive ratio in train: {(train['fire_tomorrow'] == 0).sum() / train['fire_tomorrow'].sum():.1f}")

assert train["date"].max() < test["date"].min(), "split overlaps!"
assert train[kept].isna().sum().sum() == 0 and test[kept].isna().sum().sum() == 0

print("\nDropped as redundant (|corr| > 0.9):")
for f, p in dropped.items():
    print(f"  {f}  (redundant with {p}, corr = {corr.loc[f, p]:.2f})")
print(f"\nKept {len(kept)} features:", kept)
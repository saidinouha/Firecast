from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
fires = pd.read_csv(ROOT / "data" / "processed" / "fires_tunisia_clean.csv")
fires["acq_date"] = pd.to_datetime(fires["acq_date"])

# ~1 km spots: round coordinates to 2 decimals
fires["spot"] = fires["latitude"].round(2).astype(str) + "," + fires["longitude"].round(2).astype(str)

spots = fires.groupby("spot").agg(
    days=("acq_date", "nunique"),
    years=("acq_date", lambda s: s.dt.year.nunique()),
    lat=("latitude", "mean"),
    lon=("longitude", "mean"),
    mean_frp=("frp", "mean"),
).sort_values("days", ascending=False)

print("Number of distinct spots:", len(spots))
print("\nTop 15 most persistent spots:")
print(spots.head(15))
print("\nSpots detected on 20+ different days:", (spots["days"] >= 20).sum())
print("Detections they account for:", int(fires["spot"].isin(spots[spots["days"] >= 20].index).sum()))
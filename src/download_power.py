import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "power"
OUT_DIR.mkdir(parents=True, exist_ok=True)

grid = pd.read_csv(ROOT / "data" / "processed" / "grid_cells.csv")

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = "T2M,T2M_MAX,T2M_MIN,RH2M,WS2M,PRECTOTCORR"

print(f"{len(grid)} cells to download")

for k, row in grid.iterrows():
    out_file = OUT_DIR / f"{row['cell_id']}.csv"
    if out_file.exists():              # already downloaded: skip (lets you resume)
        continue

    query = {
        "parameters": PARAMS,
        "community": "AG",
        "longitude": row["lon_c"],
        "latitude": row["lat_c"],
        "start": "20130101",
        "end": "20241231",
        "format": "JSON",
    }

    for attempt in range(3):           # retry up to 3 times
        try:
            r = requests.get(URL, params=query, timeout=120)
            r.raise_for_status()
            data = r.json()["properties"]["parameter"]
            df = pd.DataFrame(data)                    # rows = dates, columns = variables
            df.index.name = "date"
            df = df.reset_index()
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
            df.insert(0, "cell_id", row["cell_id"])
            df.to_csv(out_file, index=False)
            break
        except Exception as e:
            print(f"  cell {row['cell_id']} attempt {attempt + 1} failed: {e}")
            time.sleep(10)
    else:
        print(f"  GIVING UP on cell {row['cell_id']}, re-run the script later")

    if k % 10 == 0:
        print(f"{k + 1}/{len(grid)} done (cell {row['cell_id']})")
    time.sleep(1)                      # be polite to the server

# ---- Combine all cells into one file ----
frames = [pd.read_csv(f, parse_dates=["date"]) for f in sorted(OUT_DIR.glob("*.csv"))]
weather = pd.concat(frames, ignore_index=True)

# POWER uses -999 for missing values: convert to NaN
weather = weather.replace(-999, float("nan"))
weather.to_csv(ROOT / "data" / "raw" / "power_all.csv", index=False)

print("\nCells downloaded:", weather["cell_id"].nunique(), "of", len(grid))
print("Shape:", weather.shape, "(expected 412002 rows = 94 cells x 4383 days)")
print("Date range:", weather["date"].min().date(), "to", weather["date"].max().date())
print("\nMissing values per column:")
print(weather.isna().sum())
print("\nSummary statistics:")
print(weather.drop(columns=["cell_id", "date"]).describe().round(2))
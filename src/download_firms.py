import time
from pathlib import Path
from io import StringIO

import pandas as pd
import requests
from config import FIRMS_KEY

ROOT = Path(__file__).resolve().parent.parent
CHUNK_DIR = ROOT / "data" / "raw" / "firms_chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

AREA = "7.5,30.2,11.6,37.6"      # Tunisia bounding box
SOURCE = "VIIRS_SNPP_SP"
DAYS = 5

starts = pd.date_range("2019-01-01", "2024-12-31", freq=f"{DAYS}D")
print(f"{len(starts)} requests to make")

for i, start in enumerate(starts):
    date_str = start.strftime("%Y-%m-%d")
    out_file = CHUNK_DIR / f"{date_str}.csv"
    if out_file.exists():            # already downloaded, skip (lets you resume)
        continue

    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
           f"{FIRMS_KEY}/{SOURCE}/{AREA}/{DAYS}/{date_str}")

    for attempt in range(3):         # retry up to 3 times
        try:
            text = requests.get(url, timeout=60).text
            if "latitude" not in text:      # error message instead of CSV
                raise ValueError(text[:100])
            out_file.write_text(text)
            break
        except Exception as e:
            print(f"  {date_str} attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    else:
        print(f"  GIVING UP on {date_str}, re-run the script later")

    if i % 50 == 0:
        print(f"{i}/{len(starts)} done ({date_str})")
    time.sleep(0.3)                  # be polite to the server

# Combine all chunks into one file
frames = [pd.read_csv(f) for f in sorted(CHUNK_DIR.glob("*.csv"))]
frames = [f for f in frames if len(f) > 0]
fires = pd.concat(frames, ignore_index=True).drop_duplicates()
fires.to_csv(ROOT / "data" / "raw" / "firms_2019_2024.csv", index=False)

print("Total detections:", len(fires))
print("Date range:", fires["acq_date"].min(), "to", fires["acq_date"].max())
print(fires["type"].value_counts())
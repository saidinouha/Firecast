import time
from pathlib import Path

import pandas as pd
import requests
from config import FIRMS_KEY

ROOT = Path(__file__).resolve().parent.parent
CHUNK_DIR = ROOT / "data" / "raw" / "firms_chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

AREA = "7.5,30.2,11.6,37.6"      # Tunisia bounding box: west, south, east, north
SOURCE = "VIIRS_SNPP_SP"         # VIIRS satellite, standard archive data
DAYS = 5                         # max days per request

# Start date chosen so the 5-day steps line up with chunks already downloaded (2019-01-01)
starts = pd.date_range("2012-12-28", "2024-12-31", freq=f"{DAYS}D")
print(f"{len(starts)} chunks in total")

for i, start in enumerate(starts):
    date_str = start.strftime("%Y-%m-%d")
    out_file = CHUNK_DIR / f"{date_str}.csv"

    if out_file.exists():            # already downloaded: skip (lets you resume)
        continue

    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
           f"{FIRMS_KEY}/{SOURCE}/{AREA}/{DAYS}/{date_str}")

    for attempt in range(3):         # retry up to 3 times
        try:
            text = requests.get(url, timeout=60).text
            if "latitude" not in text:          # an error message instead of a CSV
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

# ---- Combine all chunks into one file ----
frames = [pd.read_csv(f) for f in sorted(CHUNK_DIR.glob("*.csv"))]
frames = [f for f in frames if len(f) > 0]      # skip chunks with no detections
fires = pd.concat(frames, ignore_index=True).drop_duplicates()
fires.to_csv(ROOT / "data" / "raw" / "firms_2013_2024.csv", index=False)

n_chunks = len(list(CHUNK_DIR.glob("*.csv")))
print("Chunk files:", n_chunks, "of", len(starts))
print("Total detections:", len(fires))
print("Date range:", fires["acq_date"].min(), "to", fires["acq_date"].max())
print(fires["type"].value_counts())
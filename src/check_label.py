from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "processed" / "firecast_dataset.csv.gz", parse_dates=["date"])
fires = pd.read_csv(ROOT / "data" / "processed" / "fires_tunisia_clean.csv", parse_dates=["acq_date"])

row = df[df["fire_tomorrow"] == 1].sample(1, random_state=1).iloc[0]
i, j = map(int, row["cell_id"].split("_"))
tomorrow = row["date"] + pd.Timedelta(days=1)

hits = fires[(fires["acq_date"] == tomorrow)
             & (fires["latitude"] >= i * 0.5) & (fires["latitude"] < (i + 1) * 0.5)
             & (fires["longitude"] >= j * 0.5) & (fires["longitude"] < (j + 1) * 0.5)]

print(f"Row says: cell {row['cell_id']}, date {row['date'].date()}, fire_tomorrow = 1")
print(f"Fires in that cell on {tomorrow.date()}: {len(hits)} (must be at least 1)")
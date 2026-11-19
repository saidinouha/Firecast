from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

fires = pd.read_csv(ROOT / "data" / "raw" / "firms_2013_2024.csv")
fires["acq_date"] = pd.to_datetime(fires["acq_date"])

# Keep 2013-2024 and vegetation fires only
fires = fires[(fires["acq_date"] >= "2013-01-01") & (fires["acq_date"] <= "2024-12-31")]
veg = fires[fires["type"] == 0].copy()
print("Vegetation fire detections 2013-2024:", len(veg))

print("\nDetections per year:")
print(veg["acq_date"].dt.year.value_counts().sort_index())

print("\nDetections per month (all years):")
print(veg["acq_date"].dt.month.value_counts().sort_index())

print("\nConfidence levels:")
print(veg["confidence"].value_counts())

# Map of detections: vegetation (red) vs other sources (grey)
other = fires[fires["type"] != 0]
plt.figure(figsize=(6, 8))
plt.scatter(other["longitude"], other["latitude"], s=1, c="lightgrey", label="Type 2/3 (not wildfire)")
plt.scatter(veg["longitude"], veg["latitude"], s=1, c="red", label="Type 0 (vegetation fire)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("FIRMS VIIRS detections, 2013-2024")
plt.legend(markerscale=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "fires_map.png", dpi=150)
plt.show()
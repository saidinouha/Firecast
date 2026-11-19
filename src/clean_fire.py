from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Country borders (Natural Earth, 50 m resolution)
BORDERS_URL = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"

fires = pd.read_csv(ROOT / "data" / "raw" / "firms_2013_2024.csv")
fires["acq_date"] = pd.to_datetime(fires["acq_date"])
fires = fires[(fires["acq_date"] >= "2013-01-01") & (fires["acq_date"] <= "2024-12-31")]
n_all = len(fires)

# Filter 1: vegetation fires only
fires = fires[fires["type"] == 0]
n_veg = len(fires)

# Filter 2: drop low-confidence detections
fires = fires[fires["confidence"] != "l"]
n_conf = len(fires)

# Filter 3: keep only points inside Tunisia
world = gpd.read_file(BORDERS_URL)
tunisia = world[world["ADMIN"] == "Tunisia"]
assert len(tunisia) == 1, f"Tunisia not found, columns: {world.columns.tolist()}"
border = tunisia.union_all().buffer(0.03)          # ~3 km buffer for coastal points

points = gpd.GeoDataFrame(
    fires,
    geometry=gpd.points_from_xy(fires["longitude"], fires["latitude"]),
    crs="EPSG:4326",
)
inside = points.within(border)
kept = points[inside].drop(columns="geometry")
removed = points[~inside]

print("All detections 2013-2024:      ", n_all)
print("After type 0 filter:           ", n_veg)
print("After dropping low confidence: ", n_conf)
print("After keeping Tunisia only:    ", len(kept))


# Filter 4: remove persistent fixed sources
# (same ~1 km spot detected on 20+ different days is not a wildfire)
kept = kept.copy()
kept["spot"] = kept["latitude"].round(2).astype(str) + "," + kept["longitude"].round(2).astype(str)
days_per_spot = kept.groupby("spot")["acq_date"].nunique()
fixed_spots = days_per_spot[days_per_spot >= 20].index
n_before = len(kept)
kept = kept[~kept["spot"].isin(fixed_spots)].drop(columns="spot")
print("Removed as fixed sources:      ", n_before - len(kept))
print("Final clean detections:        ", len(kept))


# Save the clean file (keep only useful columns)
cols = ["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "daynight"]
kept[cols].to_csv(OUT_DIR / "fires_tunisia_clean.csv", index=False)

# Check: detections per month for the last three years
ym = kept.groupby([kept["acq_date"].dt.year, kept["acq_date"].dt.month]).size().unstack(fill_value=0)
print("\nDetections per month:")
print(ym.loc[[2022, 2023, 2024]])

# Map: kept (red) vs removed (grey) with the border
fig, ax = plt.subplots(figsize=(6, 8))
tunisia.boundary.plot(ax=ax, color="black", linewidth=0.8)
ax.scatter(removed["longitude"], removed["latitude"], s=1, c="lightgrey", label="Removed (outside Tunisia)")
ax.scatter(kept["longitude"], kept["latitude"], s=1, c="red", label="Kept")
ax.set_xlim(7.5, 11.6)
ax.set_ylim(30.2, 37.6)
ax.set_title("Clean vegetation fire detections, 2013-2024")
ax.legend(markerscale=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "fires_clean_map.png", dpi=150)
plt.show()
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
BORDERS_URL = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
STEP = 0.5   # cell size in degrees

world = gpd.read_file(BORDERS_URL)
tunisia = world[world["ADMIN"] == "Tunisia"]
border = tunisia.union_all().buffer(0.03)    # same buffer as in clean_fires.py

# Candidate cells covering the bounding box, indexed by integers
rows = []
for i in range(int(30.0 / STEP), int(37.5 / STEP)):        # latitude index
    for j in range(int(7.5 / STEP), int(12.0 / STEP)):     # longitude index
        cell = box(j * STEP, i * STEP, (j + 1) * STEP, (i + 1) * STEP)
        inter = cell.intersection(border).area / cell.area
        if inter > 0:
            rows.append({
                "cell_id": f"{i}_{j}",
                "lat_idx": i, "lon_idx": j,
                "lat_c": (i + 0.5) * STEP,
                "lon_c": (j + 0.5) * STEP,
                "frac_inside": round(inter, 3),
            })
grid = pd.DataFrame(rows)

# Assign each clean fire to its cell and count
fires = pd.read_csv(ROOT / "data" / "processed" / "fires_tunisia_clean.csv")
fires["lat_idx"] = np.floor(fires["latitude"] / STEP).astype(int)
fires["lon_idx"] = np.floor(fires["longitude"] / STEP).astype(int)
fires["cell_id"] = fires["lat_idx"].astype(str) + "_" + fires["lon_idx"].astype(str)

counts = fires.groupby("cell_id").size().rename("n_fires")
grid = grid.merge(counts, on="cell_id", how="left").fillna({"n_fires": 0})
grid["n_fires"] = grid["n_fires"].astype(int)

# Safety check: every fire must belong to a cell of the grid
missing = set(fires["cell_id"]) - set(grid["cell_id"])
print("Fires in cells missing from the grid:", len(missing))

grid.to_csv(ROOT / "data" / "processed" / "grid_cells.csv", index=False)

print("Number of cells:", len(grid))
print("Cells with zero fires in 12 years:", (grid["n_fires"] == 0).sum())
print("\nFires per cell (summary):")
print(grid["n_fires"].describe())
print("\nTop 10 cells:")
print(grid.sort_values("n_fires", ascending=False).head(10)[["cell_id", "lat_c", "lon_c", "frac_inside", "n_fires"]])

# Map: grid over the border, colored by fire count
fig, ax = plt.subplots(figsize=(6, 8))
tunisia.boundary.plot(ax=ax, color="black", linewidth=0.8)
for _, r in grid.iterrows():
    ax.add_patch(plt.Rectangle((r["lon_idx"] * STEP, r["lat_idx"] * STEP), STEP, STEP,
                               fill=False, edgecolor="grey", linewidth=0.4))
sc = ax.scatter(grid["lon_c"], grid["lat_c"], c=grid["n_fires"], s=60, cmap="YlOrRd")
plt.colorbar(sc, label="Fire detections 2013-2024")
ax.set_xlim(7.5, 12.0)
ax.set_ylim(30.0, 37.5)
ax.set_title("Grid cells (0.5°) colored by fire count")
plt.tight_layout()
plt.savefig(FIG_DIR / "grid_map.png", dpi=150)
plt.show()
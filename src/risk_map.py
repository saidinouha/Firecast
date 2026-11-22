import json
from pathlib import Path

import joblib
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

ROOT = Path(__file__).resolve().parent.parent
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
FIG_DIR = RES / "figures"
STEP = 0.5
BORDERS_URL = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"

test = pd.read_csv(PROC / "test.csv.gz", parse_dates=["date"])
grid = pd.read_csv(PROC / "grid_cells.csv")
fires = pd.read_csv(PROC / "fires_tunisia_clean.csv", parse_dates=["acq_date"])
features = json.load(open(PROC / "selected_features.json"))["selected"]
thr_high = json.load(open(RES / "thresholds.json"))["Random Forest"]
model = joblib.load(RES / "models" / "rf.joblib")

world = gpd.read_file(BORDERS_URL)
tunisia = world[world["ADMIN"] == "Tunisia"]

COLORS = {"low": "#7bc67b", "medium": "#ffb347", "high": "#e34a33"}


def level(p):
    return "high" if p >= thr_high else ("medium" if p >= 0.5 else "low")


def draw(date, filename):
    date = pd.Timestamp(date)
    day = test[test["date"] == date].merge(grid[["cell_id", "lat_idx", "lon_idx"]], on="cell_id")
    assert len(day) == len(grid), f"expected {len(grid)} cells, got {len(day)}"
    day["score"] = model.predict_proba(day[features])[:, 1]
    day["level"] = day["score"].map(level)

    tomorrow = date + pd.Timedelta(days=1)
    todays_fires = fires[fires["acq_date"] == tomorrow]

    print(f"\n=== Prediction made on {date.date()} for {tomorrow.date()} ===")
    print("Cells per risk level:", day["level"].value_counts().to_dict())
    print("Real fires that day:", len(todays_fires), "detections in",
          int(day["fire_tomorrow"].sum()), "cells")
    print("Risk level of the cells that really had a fire:",
          day.loc[day["fire_tomorrow"] == 1, "level"].value_counts().to_dict())

    fig, ax = plt.subplots(figsize=(6.5, 8.5))
    for _, r in day.iterrows():
        ax.add_patch(Rectangle((r["lon_idx"] * STEP, r["lat_idx"] * STEP), STEP, STEP,
                               facecolor=COLORS[r["level"]], edgecolor="white",
                               linewidth=0.6, alpha=0.75))
    tunisia.boundary.plot(ax=ax, color="black", linewidth=0.9)
    ax.scatter(todays_fires["longitude"], todays_fires["latitude"],
               s=14, c="black", marker="o", label="Real fire detections")
    ax.set_xlim(7.5, 12.0)
    ax.set_ylim(30.0, 37.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Predicted fire risk for {tomorrow.date()}\n(made on {date.date()}, Random Forest)")
    handles = [Patch(facecolor=COLORS[k], alpha=0.75, label=f"{k.capitalize()} risk") for k in COLORS]
    handles.append(plt.Line2D([], [], marker="o", color="black", linestyle="", label="Real fire detections"))
    ax.legend(handles=handles, loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150)
    plt.show()


# Busiest test day (most cells with a fire the next day) and a winter day for contrast
peak_day = test.groupby("date")["fire_tomorrow"].sum().idxmax()
draw(peak_day, "risk_map_peak.png")
draw("2023-01-15", "risk_map_winter.png")
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
K = 12   # keep a feature if it is in the top K of at least one method

rank = pd.read_csv(ROOT / "results" / "feature_ranking.csv", index_col=0)
order = json.load(open(PROC / "features.json"))["features"]     # original order

selected = [f for f in order if rank.loc[f, "rf_rank"] <= K or rank.loc[f, "mi_rank"] <= K]
dropped = [f for f in order if f not in selected]

with open(PROC / "selected_features.json", "w") as f:
    json.dump({"selected": selected, "dropped_low_rank": dropped}, f, indent=2)

print(f"Selected {len(selected)} features:", selected)
print("Dropped:", dropped)

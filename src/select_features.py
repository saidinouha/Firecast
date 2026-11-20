import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(PROC / "train.csv.gz", parse_dates=["date"])
features = json.load(open(PROC / "features.json"))["features"]
X, y = train[features], train["fire_tomorrow"]

# Method 1: Random Forest importance
rf = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_leaf=50,
    class_weight="balanced_subsample", n_jobs=-1, random_state=42,
)
rf.fit(X, y)
imp_rf = pd.Series(rf.feature_importances_, index=features)

# Method 2: mutual information (measures any kind of dependence, not just straight lines)
sample = train.sample(80000, random_state=42)
mi = mutual_info_classif(sample[features], sample["fire_tomorrow"], random_state=42)
imp_mi = pd.Series(mi, index=features)

table = pd.DataFrame({"rf_importance": imp_rf, "mutual_info": imp_mi})
table["rf_rank"] = table["rf_importance"].rank(ascending=False).astype(int)
table["mi_rank"] = table["mutual_info"].rank(ascending=False).astype(int)
table = table.sort_values("rf_importance", ascending=False)
print(table.round(4))

table.to_csv(ROOT / "results" / "feature_ranking.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
sns.barplot(x=table["rf_importance"], y=table.index, ax=axes[0], color="tomato")
axes[0].set_title("Random Forest importance")
mi_sorted = table["mutual_info"].sort_values(ascending=False)
sns.barplot(x=mi_sorted.values, y=mi_sorted.index, ax=axes[1], color="steelblue")
axes[1].set_title("Mutual information")
plt.tight_layout()
plt.savefig(FIG_DIR / "feature_selection.png", dpi=150)
plt.show()
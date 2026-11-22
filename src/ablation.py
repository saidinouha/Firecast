import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
FIG_DIR, TAB_DIR = RES / "figures", RES / "tables"

train = pd.read_csv(PROC / "train.csv.gz", parse_dates=["date"])
test = pd.read_csv(PROC / "test.csv.gz", parse_dates=["date"])
features = json.load(open(PROC / "selected_features.json"))["selected"]
best = json.load(open(RES / "best_params.json"))
y_train, y_test = train["fire_tomorrow"], test["fire_tomorrow"]
base_rate = y_test.mean()

groups = {
    "fire history": ["fires_today_log", "fires_last3_log", "fires_last7_log", "fires_last30_log"],
    "location": ["lat_c", "lon_c", "frac_inside"],
    "season": ["doy_sin", "doy_cos"],
    "weather": ["temp_max", "temp_min", "humidity", "rain_30d_log"],
}
assert sorted(sum(groups.values(), [])) == sorted(features), \
    f"groups do not match selected features: {features}"


def without(*names):
    return [f for g, fs in groups.items() if g not in names for f in fs]


def only(*names):
    return [f for g in names for f in groups[g]]


variants = {
    "All 13 features": features,
    "Without fire history": without("fire history"),
    "Without location": without("location"),
    "Without weather": without("weather"),
    "Without season": without("season"),
    "Fire history only": only("fire history"),
    "Weather + season only": only("weather", "season"),
}


def make_lr():
    return Pipeline([("scale", StandardScaler()),
                     ("model", LogisticRegression(class_weight="balanced", max_iter=1000,
                                                  C=best["logreg"]["model__C"]))])


def make_rf():
    return RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=42, **best["rf"])


rows = []
for vname, cols in variants.items():
    for mname, make in [("Logistic Regression", make_lr), ("Random Forest", make_rf)]:
        model = make().fit(train[cols], y_train)
        p = model.predict_proba(test[cols])[:, 1]
        pr = average_precision_score(y_test, p)
        rows.append({"variant": vname, "model": mname, "n_features": len(cols),
                     "PR_AUC": pr, "ROC_AUC": roc_auc_score(y_test, p),
                     "lift_vs_random": pr / base_rate})
        print(f"{vname:24s} {mname:20s} PR-AUC {pr:.4f}")

res = pd.DataFrame(rows)
res.to_csv(TAB_DIR / "ablation.csv", index=False)

pd.set_option("display.width", 200)
wide = res.pivot(index="variant", columns="model", values="PR_AUC").loc[list(variants)]
wide["RF lift vs random"] = wide["Random Forest"] / base_rate
print(f"\n=== Ablation, test 2022-2024 (random model PR-AUC = {base_rate:.4f}) ===")
print(wide.round(4))

plt.figure(figsize=(9, 5))
sns.barplot(data=res, y="variant", x="PR_AUC", hue="model", order=list(variants))
plt.axvline(base_rate, color="grey", linestyle="--", label="Random model")
plt.legend()
plt.title("Ablation: test PR-AUC with different feature groups")
plt.tight_layout()
plt.savefig(FIG_DIR / "ablation.png", dpi=150)
plt.show()
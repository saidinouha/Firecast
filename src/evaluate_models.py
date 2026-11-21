import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
FIG_DIR, TAB_DIR = RES / "figures", RES / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(PROC / "train.csv.gz", parse_dates=["date"])
test = pd.read_csv(PROC / "test.csv.gz", parse_dates=["date"])
features = json.load(open(PROC / "selected_features.json"))["selected"]
best = json.load(open(RES / "best_params.json"))
preds = pd.read_csv(RES / "test_predictions.csv.gz", parse_dates=["date"])
assert (preds["y_true"].values == test["fire_tomorrow"].values).all(), "prediction file misaligned"

X, y = train[features], train["fire_tomorrow"]
y_test = test["fire_tomorrow"].values
scores = {"Logistic Regression": preds["p_logreg"].values, "Random Forest": preds["p_rf"].values}

# Models rebuilt with the best parameters (used only to get out-of-fold scores)
models = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, C=best["logreg"]["model__C"]))]),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced_subsample", n_jobs=-1, random_state=42, **best["rf"]),
}

# ---- 1. Choose the threshold on out-of-fold predictions of the TRAINING years ----
cv = TimeSeriesSplit(n_splits=4)


def oof_scores(model):
    idx, sc = [], []
    for tr, va in cv.split(X):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        idx.append(va)
        sc.append(m.predict_proba(X.iloc[va])[:, 1])
    idx, sc = np.concatenate(idx), np.concatenate(sc)
    return y.iloc[idx].values, sc


def best_f1_threshold(y_true, score):
    p, r, t = precision_recall_curve(y_true, score)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(t[np.argmax(f1)])


thresholds = {}
for name, model in models.items():
    yt, sc = oof_scores(model)
    thresholds[name] = best_f1_threshold(yt, sc)
    print(f"{name}: threshold chosen on training folds = {thresholds[name]:.3f}")
json.dump(thresholds, open(RES / "thresholds.json", "w"), indent=2)

# ---- 2. Ranking quality on the test set (threshold-free) ----
month_rate = train.groupby(train["date"].dt.month)["fire_tomorrow"].mean()
rank_scores = {
    "Baseline A: monthly rate": test["date"].dt.month.map(month_rate).values,
    "Baseline B': fire count last 30 days": test["fires_last30_log"].values,
    "Logistic Regression": scores["Logistic Regression"],
    "Random Forest": scores["Random Forest"],
}
rank_table = pd.DataFrame([
    {"model": n, "PR_AUC": average_precision_score(y_test, s), "ROC_AUC": roc_auc_score(y_test, s)}
    for n, s in rank_scores.items()])
print("\n=== Ranking quality, test 2022-2024 (random model PR-AUC = %.4f) ===" % y_test.mean())
print(rank_table.round(4).to_string(index=False))
rank_table.to_csv(TAB_DIR / "final_ranking_metrics.csv", index=False)

# ---- 3. Decisions at a threshold + confusion matrices ----
def decision_row(label, pred):
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {"model": label, "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred), "F1": f1_score(y_test, pred),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}, np.array([[tn, fp], [fn, tp]])


rows, cms = [], {}
pred_b = (test["fires_last7_log"] > 0).astype(int).values
r, cm_b = decision_row("Baseline B: fire in last 7 days", pred_b)
rows.append(r)
for name in models:
    for tag, thr in [("default", 0.5), ("tuned", thresholds[name])]:
        label = f"{name}, thr {thr:.2f}" + (" (tuned)" if tag == "tuned" else "")
        r, cm = decision_row(label, (scores[name] >= thr).astype(int))
        rows.append(r)
        cms[(name, tag)] = cm
dec_table = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print("\n=== Decisions on the test set ===")
print(dec_table.round(4).to_string(index=False))
dec_table.to_csv(TAB_DIR / "final_decision_metrics.csv", index=False)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for i, name in enumerate(models):
    for j, (tag, thr) in enumerate([("default", 0.5), ("tuned", thresholds[name])]):
        cm = cms[(name, tag)]
        sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=cm, fmt="d", cmap="Blues",
                    cbar=False, ax=axes[i, j], vmin=0, vmax=1,
                    xticklabels=["No fire", "Fire"], yticklabels=["No fire", "Fire"])
        axes[i, j].set_title(f"{name}\nthreshold = {thr:.2f}")
        axes[i, j].set_xlabel("Predicted")
        axes[i, j].set_ylabel("Actual")
plt.suptitle("Confusion matrices, test 2022-2024 (colour = share of each actual class)")
plt.tight_layout()
plt.savefig(FIG_DIR / "confusion_matrices.png", dpi=150)
plt.show()

# ---- 4. Precision-recall curves ----
plt.figure(figsize=(7, 5))
for n, s in rank_scores.items():
    p, r_, _ = precision_recall_curve(y_test, s)
    plt.plot(r_, p, label=f"{n} (PR-AUC {average_precision_score(y_test, s):.3f})")
plt.axhline(y_test.mean(), color="grey", linestyle="--", label="Random model")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-recall curves, test 2022-2024")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "pr_curves.png", dpi=150)
plt.show()

# ---- 5. Test results per year ----
yr = []
for year in sorted(test["date"].dt.year.unique()):
    m = (test["date"].dt.year == year).values
    yr.append({"year": year, "positive_rate": y_test[m].mean(),
               "PR_AUC_LogReg": average_precision_score(y_test[m], scores["Logistic Regression"][m]),
               "PR_AUC_RF": average_precision_score(y_test[m], scores["Random Forest"][m])})
yr = pd.DataFrame(yr)
print("\n=== Test results per year ===")
print(yr.round(4).to_string(index=False))
yr.to_csv(TAB_DIR / "final_per_year.csv", index=False)
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
(RES / "tables").mkdir(parents=True, exist_ok=True)
(RES / "models").mkdir(parents=True, exist_ok=True)

train = pd.read_csv(PROC / "train.csv.gz", parse_dates=["date"])
test = pd.read_csv(PROC / "test.csv.gz", parse_dates=["date"])
features = json.load(open(PROC / "selected_features.json"))["selected"]
assert train["date"].is_monotonic_increasing, "train must be sorted by date"

X_train, y_train = train[features], train["fire_tomorrow"]
X_test, y_test = test[features], test["fire_tomorrow"]
print(f"{len(features)} features, train {len(train)} rows, test {len(test)} rows")
print(f"Random-model PR-AUC reference (test positive rate): {y_test.mean():.4f}")

cv = TimeSeriesSplit(n_splits=4)     # 4 folds, each validated on a period after its training data


def evaluate(name, y_true, score, pred=None):
    row = {"model": name,
           "PR_AUC": average_precision_score(y_true, score),
           "ROC_AUC": roc_auc_score(y_true, score)}
    if pred is not None:
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        row.update(precision=precision_score(y_true, pred, zero_division=0),
                   recall=recall_score(y_true, pred), F1=f1_score(y_true, pred),
                   TP=tp, FP=fp, FN=fn, TN=tn)
    return row


def run_search(name, estimator, grid, n_jobs, verbose):
    start = time.time()
    search = GridSearchCV(estimator, grid, scoring="average_precision",
                          cv=cv, n_jobs=n_jobs, verbose=verbose)
    search.fit(X_train, y_train)
    print(f"\n{name}: best params {search.best_params_}, "
          f"CV PR-AUC {search.best_score_:.4f}  ({time.time() - start:.0f} s)")
    cvr = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    print(cvr.sort_values("rank_test_score").round(4).to_string(index=False))
    cvr.to_csv(RES / "tables" / f"cv_results_{name}.csv", index=False)
    return search


results, preds = [], pd.DataFrame({"cell_id": test["cell_id"], "date": test["date"], "y_true": y_test})

# ---- Baseline A: monthly fire rate learned on the training years ----
monthly_rate = train.groupby(train["date"].dt.month)["fire_tomorrow"].mean()
score_month = test["date"].dt.month.map(monthly_rate).values
results.append(evaluate("Baseline A: monthly rate", y_test, score_month))
preds["baseline_month"] = score_month

# ---- Baseline B: any fire in this cell in the last 7 days ----
pred_last7 = (test["fires_last7_log"] > 0).astype(int).values
results.append(evaluate("Baseline B: fire in last 7 days", y_test, pred_last7, pred_last7))
preds["baseline_last7"] = pred_last7

# ---- Logistic Regression ----
lr_pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
])
lr_search = run_search("logreg", lr_pipe, {"model__C": [0.01, 0.1, 1, 10]}, n_jobs=-1, verbose=1)
p_lr = lr_search.best_estimator_.predict_proba(X_test)[:, 1]
results.append(evaluate("Logistic Regression", y_test, p_lr, (p_lr >= 0.5).astype(int)))
preds["p_logreg"] = p_lr

# ---- Random Forest ----
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample",
                            n_jobs=-1, random_state=42)
rf_grid = {"max_depth": [6, 12], "min_samples_leaf": [20, 100]}
rf_search = run_search("rf", rf, rf_grid, n_jobs=1, verbose=2)
p_rf = rf_search.best_estimator_.predict_proba(X_test)[:, 1]
results.append(evaluate("Random Forest", y_test, p_rf, (p_rf >= 0.5).astype(int)))
preds["p_rf"] = p_rf

# ---- Save everything needed for Step 8 ----
table = pd.DataFrame(results)
table.to_csv(RES / "tables" / "test_results_step7.csv", index=False)
preds.to_csv(RES / "test_predictions.csv.gz", index=False)
json.dump({"logreg": lr_search.best_params_, "rf": rf_search.best_params_},
          open(RES / "best_params.json", "w"), indent=2)
joblib.dump(lr_search.best_estimator_, RES / "models" / "logreg.joblib", compress=3)
joblib.dump(rf_search.best_estimator_, RES / "models" / "rf.joblib", compress=3)

pd.set_option("display.width", 200)
print("\n===== RESULTS ON THE TEST SET (2022-2024), threshold 0.5 =====")
print(table.round(4).to_string(index=False))
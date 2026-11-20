from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "data" / "processed" / "firecast_dataset.csv.gz", parse_dates=["date"])
sns.set_theme(style="whitegrid")

# ---- 1. Display the data ----
print(df.head())
df.info()
print(df.describe().round(2).T)
print("Duplicate cell-days:", df.duplicated(["cell_id", "date"]).sum())

# ---- 2. Class balance ----
plt.figure(figsize=(4, 4))
ax = sns.countplot(data=df, x="fire_tomorrow")
ax.bar_label(ax.containers[0])
ax.set_title("Class balance")
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_class_balance.png", dpi=150)
plt.show()

# ---- 3. Fire rate per month ----
rate = df.groupby("month")["fire_tomorrow"].mean() * 100
plt.figure(figsize=(7, 4))
sns.barplot(x=rate.index, y=rate.values, color="tomato")
plt.xlabel("Month")
plt.ylabel("% of cell-days with a fire the next day")
plt.title("Fire seasonality")
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_monthly_rate.png", dpi=150)
plt.show()

# ---- 4. Distributions of the main variables ----
cols = ["temp_mean", "humidity", "wind", "rain", "rain_30d", "fires_last30"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, c in zip(axes.ravel(), cols):
    sns.histplot(df[c], bins=50, ax=ax)
    if c in ("rain", "rain_30d", "fires_last30"):
        ax.set_yscale("log")           # most values are 0: log scale shows the tail
    ax.set_title(c)
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_distributions.png", dpi=150)
plt.show()

# ---- 5. Features vs. label (boxplots) ----
cols = ["temp_max", "humidity", "wind", "rain_30d"]
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, c in zip(axes, cols):
    sns.boxplot(data=df, x="fire_tomorrow", y=c, ax=ax)
    ax.set_title(c)
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_boxplots.png", dpi=150)
plt.show()

# ---- 6. Outlier check (IQR rule) ----
num = ["temp_mean", "temp_max", "temp_min", "humidity", "wind", "rain",
       "rain_7d", "rain_30d", "temp_7d", "humidity_7d"]
rows = []
for c in num:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum()
    rows.append({"variable": c, "outliers": n_out, "percent": round(100 * n_out / len(df), 2)})
print("\nOutliers by the IQR rule:")
print(pd.DataFrame(rows).to_string(index=False))

# ---- 7. Correlation heatmap ----
corr = df[num + ["fires_today", "fires_last3", "fires_last7", "fires_last30",
                 "month", "fire_tomorrow"]].corr()
plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7})
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_correlation.png", dpi=150)
plt.show()
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "processed" / "firecast_dataset.csv.gz", parse_dates=["date"])

print("Rain percentiles (mm/day):")
print(df["rain"].quantile([0.5, 0.9, 0.99, 0.999, 0.9999]).round(2))

print("\nCell-days with rain > 50 mm: ", (df["rain"] > 50).sum())
print("Cell-days with rain > 100 mm:", (df["rain"] > 100).sum())
print("Cell-days with rain > 200 mm:", (df["rain"] > 200).sum())

print("\nTop 10 rain values:")
print(df.nlargest(10, "rain")[["cell_id", "date", "rain", "humidity", "fire_tomorrow"]])

print("\nDates with the most extreme rain (cells above 100 mm that day):")
big = (df[df["rain"] > 100].groupby("date")
       .agg(n_cells=("cell_id", "nunique"), max_rain=("rain", "max"))
       .sort_values("max_rain", ascending=False))
print(big.head(10))

print("\nCell 74_18 around the top event:")
sub = df[(df["cell_id"] == "74_18") & df["date"].between("2023-12-14", "2023-12-22")]
print(sub[["date", "rain", "humidity", "temp_mean"]].to_string(index=False))
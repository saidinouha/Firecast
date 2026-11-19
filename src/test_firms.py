from pathlib import Path
import pandas as pd
from config import FIRMS_KEY


ROOT = Path(__file__).resolve().parent.parent   # the FireCast folder
# Bounding box of Tunisia: west, south, east, north
area = "7.5,30.2,11.6,37.6"
source = "VIIRS_SNPP_SP"      # VIIRS satellite, standard archive data
days = 5
start_date = "2023-07-20"

url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
       f"{FIRMS_KEY}/{source}/{area}/{days}/{start_date}")

df = pd.read_csv(url)
print("Shape:", df.shape)
print(df.columns.tolist())
print(df.head())

out_path = ROOT / "data" / "raw" / "firms_test.csv"
df.to_csv(out_path, index=False)
print("Saved to:", out_path)
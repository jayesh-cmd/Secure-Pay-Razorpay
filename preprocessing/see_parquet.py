import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
pq = pd.read_parquet(ROOT / "features.parquet")
print(pq.head())
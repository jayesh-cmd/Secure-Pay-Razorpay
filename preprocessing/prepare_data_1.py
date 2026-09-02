import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
csv_path = ROOT / "fraud_detection.csv"
use_cols = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest','isFraud']

df = pd.read_csv(csv_path, usecols=use_cols)
df = df.drop_duplicates().sort_values("step").reset_index(drop=True)

for c in ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
    if c in df.columns:
        df[c]=df[c].clip(lower=0)

print(df.head())
print(df.isFraud.mean())
df.to_parquet(ROOT / "prepared_txn", index=False)

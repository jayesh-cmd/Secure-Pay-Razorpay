import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
pq_file_path = ROOT / "prepared_txn"
output_path = ROOT / "features.parquet"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['log_amount'] = np.log1p(df['amount']) # To Shrink The Larger Amounts, Keep All Amounts In Range instead of (50000 vs 10) - (10.8, 2.3)

    df['prev_step_user'] = df.groupby("nameOrig")["step"].shift(1) # Get The Time Of Last Transaction
    df['recency_hours'] = (df['step'] - df['prev_step_user']).fillna(1e6) # Get The Time Difference Between Current And Last Transaction
    df['txn_count_24h'] = df.groupby('nameOrig')['step'].diff().lt(24).groupby(df['nameOrig']).cumsum().fillna(0)

    df['user_dest_count'] = (
        df.groupby(['nameOrig','nameDest']).cumcount()
        )
    

    df['is_dest_new'] = (df['user_dest_count']==0).astype(int)
    df = df.drop(columns=['user_dest_count'])

    df['hours_day'] = df['step']%24

    dummies = pd.get_dummies(df['type'], prefix='type')
    df = pd.concat([df, dummies], axis=1)

    label = 'isFraud'
    new_cols = [
        'step', 'nameOrig', 'nameDest', 'amount', 'log_amount', 'recency_hours', 'txn_count_24h', 'is_dest_new', 'hours_day', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'
    ] + list(dummies.columns) + [label]

    return df[new_cols]

df = pd.read_parquet(pq_file_path)
df_feat = build_features(df)
# Tiny NA cleanup
df_feat = df_feat.fillna(0)
df_feat.to_parquet(output_path, index=False)
print("Saved ->", output_path, "Rows:", len(df_feat))
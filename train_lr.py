import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
train = pd.read_parquet(ROOT / "splitted_DS" / "train.parquet")
val   = pd.read_parquet(ROOT / "splitted_DS" / "val.parquet")
test  = pd.read_parquet(ROOT / "splitted_DS" / "test.parquet")

features = [
    "amount", "log_amount", "recency_hours", "txn_count_24h", "is_dest_new", "hours_day",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"
]

X_train = train[features]
y_train = train['isFraud']

X_val = val[features]
y_val = val['isFraud']

X_test = test[features]
y_test = test['isFraud']

scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()
model = xgb.XGBClassifier(
    n_estimators = 100,
    max_depth = 6,
    learning_rate = 0.1,
    scale_pos_weight = scale_pos_weight,
    random_state = 42
)
model.fit(X_train,y_train)

def evaluate(X, y, name):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred_08 = (y_prob >= 0.8).astype(int)
    
    print(f"\n{name}:")
    print(f"  AUC: {roc_auc_score(y, y_prob):.4f}")
    print(f"  Precision: {precision_score(y, y_pred_08, zero_division=0):.4f}")
    print(f"  Recall: {recall_score(y, y_pred_08):.4f}")

evaluate(X_train, y_train, "Train")
evaluate(X_val, y_val, "Val")
evaluate(X_test, y_test, "Test")

# coefficients = model.coef_[0]
# importance_df = pd.DataFrame({
#     'feature': features,
#     'coefficient': coefficients,
#     'abs_coef': np.abs(coefficients)
# }).sort_values('abs_coef', ascending=False)

# print("\n🔍 Top 10 Most Important Features:")
# for i, row in importance_df.head(10).iterrows():
#     direction = "increases" if row['coefficient'] > 0 else "decreases"
#     print(f"  {row['feature']:<20} {row['coefficient']:>8.4f}  ({direction} fraud risk)")

import joblib
from pathlib import Path

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/xboost_model.pkl")
print("\n✅ Model saved!")

# Test the saved model immediately
print("\n🧪 Testing saved model...")
loaded_model = joblib.load("models/xboost_model.pkl")
test_prob = loaded_model.predict_proba(X_test.head(1))
print(f"Sample prediction: {test_prob}")
print(f"Actual label: {y_test.iloc[0]}")
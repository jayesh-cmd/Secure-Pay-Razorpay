"""
combine_risk.py
===============
Joins the XGBoost model's per-transaction fraud score with the graph-based
ring score from ring_detection.py.

For every transaction we produce:
  fraud_score      — model.predict_proba()[:, 1]  (XGBoost)
  ring_score       — max(sender_ring, receiver_ring) from detect_rings()
  final_risk_score — combined signal (see formula below)

Combination formula
-------------------
    final_risk_score = fraud_score + (1 - fraud_score) * ring_score * 0.40

Intuition: ring evidence can "push up" the remaining clean probability by up
to 40 percentage points.  A transaction already flagged by the model (high
fraud_score) changes very little; a borderline or missed transaction with a
high ring_score gets a meaningful bump.

Examples:
  fraud=0.05, ring=1.0  →  0.05 + 0.95*0.40 = 0.43   (borderline → medium risk)
  fraud=0.30, ring=1.0  →  0.30 + 0.70*0.40 = 0.58   (weak signal → medium-high)
  fraud=0.90, ring=0.0  →  0.90 + 0.10*0.00 = 0.90   (model already sure, ring adds nothing)
  fraud=0.90, ring=1.0  →  0.90 + 0.10*0.40 = 0.94   (model sure, tiny nudge)

Output: combined_risk.parquet
  Columns: step, nameOrig, nameDest, amount, isFraud,
            fraud_score, sender_ring_score, receiver_ring_score,
            ring_score, final_risk_score
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from ring_detection import detect_rings   # reuse the function directly

ROOT = Path(__file__).resolve().parent

FEATURES = [
    "amount", "log_amount", "recency_hours", "txn_count_24h", "is_dest_new", "hours_day",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
]

RING_WEIGHT = 0.40   # max fraction of clean probability ring evidence can claim


# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading val.parquet and test.parquet...")
val  = pd.read_parquet(ROOT / "splitted_DS" / "val.parquet")
test = pd.read_parquet(ROOT / "splitted_DS" / "test.parquet")
df   = pd.concat([val, test], ignore_index=True)
print(f"  {len(df):,} transactions loaded\n")


# ── 2. Get XGBoost fraud scores ───────────────────────────────────────────────
print("Loading XGBoost model and scoring transactions...")
model       = joblib.load(ROOT / "models" / "xboost_model.pkl")
fraud_score = model.predict_proba(df[FEATURES])[:, 1]
df["fraud_score"] = np.round(fraud_score, 6)
print(f"  fraud_score  min={fraud_score.min():.4f}  "
      f"max={fraud_score.max():.4f}  "
      f"mean={fraud_score.mean():.4f}\n")


# ── 3. Run ring detection ─────────────────────────────────────────────────────
print("Running ring detection (detect_rings)...")
ring_results = detect_rings(df)   # returns: account_id, ring_score, pattern, detail
# Build a fast lookup dict: account_id → ring_score
ring_lookup = dict(zip(ring_results["account_id"], ring_results["ring_score"]))
print(f"  {len(ring_lookup):,} accounts have a ring_score\n")


# ── 4. Join ring scores onto every transaction ────────────────────────────────
# Each transaction has two accounts: sender (nameOrig) and receiver (nameDest).
# We look up ring_score for both and take the maximum as the transaction's ring_score.
df["sender_ring_score"]   = df["nameOrig"].map(ring_lookup).fillna(0.0)
df["receiver_ring_score"] = df["nameDest"].map(ring_lookup).fillna(0.0)
df["ring_score"]          = np.maximum(df["sender_ring_score"], df["receiver_ring_score"])
df["ring_score"]          = df["ring_score"].round(6)

covered = (df["ring_score"] > 0).sum()
print(f"Transactions with ring_score > 0: {covered:,} "
      f"({covered/len(df)*100:.2f}% of total)\n")


# ── 5. Compute final_risk_score ───────────────────────────────────────────────
# formula: fraud_score + (1 - fraud_score) * ring_score * RING_WEIGHT
df["final_risk_score"] = (
    df["fraud_score"] + (1 - df["fraud_score"]) * df["ring_score"] * RING_WEIGHT
).clip(0.0, 1.0).round(6)


# ── 6. Save ───────────────────────────────────────────────────────────────────
keep_cols = [
    "step", "nameOrig", "nameDest", "amount", "isFraud",
    "fraud_score", "sender_ring_score", "receiver_ring_score",
    "ring_score", "final_risk_score",
]
out = df[keep_cols].copy()
out_path = ROOT / "combined_risk.parquet"
out.to_parquet(out_path, index=False)
print(f"Saved → {out_path}  ({len(out):,} rows, {len(keep_cols)} columns)\n")


# ── 7. Summary stats ──────────────────────────────────────────────────────────
print("=" * 70)
print("  SCORE COMPARISON SUMMARY")
print("=" * 70)

thresholds = [0.4, 0.5, 0.7, 0.8]
print(f"\n  {'Threshold':<12}  {'fraud_score':>13}  {'final_risk_score':>17}  {'Δ flagged':>10}")
print("  " + "-" * 58)
for t in thresholds:
    n_fraud = (df["fraud_score"]      >= t).sum()
    n_final = (df["final_risk_score"] >= t).sum()
    delta   = n_final - n_fraud
    print(f"  ≥ {t:<9}  {n_fraud:>13,}  {n_final:>17,}  {delta:>+10,}")

# Transactions where ring pushes a sub-threshold transaction over 0.5
boost_cases = df[
    (df["fraud_score"] < 0.5) &
    (df["final_risk_score"] >= 0.5) &
    (df["ring_score"] > 0)
]
print(f"\n  Ring score lifted {len(boost_cases):,} transactions from below 0.5 "
      f"to ≥ 0.5 final_risk_score")

# ── 8. Print 5 example rows ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  5 EXAMPLE ROWS (mix of ring-boosted and plain fraud)")
print("=" * 70)

# Pick 2 ring-boosted, 2 high-fraud-only, 1 clean
sample_ring   = df[df["ring_score"] > 0].nlargest(2, "ring_score")
sample_fraud  = df[df["ring_score"] == 0].nlargest(2, "fraud_score")
sample_clean  = df[df["fraud_score"] < 0.1].sample(1, random_state=42)
sample        = pd.concat([sample_ring, sample_fraud, sample_clean])

print(f"\n  {'account_id (nameOrig)':<22} {'fraud_score':>12} {'ring_score':>11} {'final_risk':>11}  isFraud")
print("  " + "-" * 68)
for _, r in sample.iterrows():
    tag = " ⚠" if r["isFraud"] else ""
    print(f"  {r['nameOrig']:<22}  {r['fraud_score']:>10.4f}  "
          f"{r['ring_score']:>10.4f}  {r['final_risk_score']:>10.4f}  {int(r['isFraud'])}{tag}")

print()

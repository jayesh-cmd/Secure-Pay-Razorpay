from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import joblib
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import re
from langchain_groq import ChatGroq
import os
import json
import threading
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path

# ── import ring detection from project root ───────────────────────────────────
import sys
MODELS_DIR = Path(__file__).resolve().parent
ROOT       = MODELS_DIR.parent
sys.path.insert(0, str(ROOT))
from ring_detection import detect_rings, FANIN_THRESHOLD, CHAIN_MIN_LEN

load_dotenv()

app = FastAPI(title="Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Load model ────────────────────────────────────────────────────────────────
model = joblib.load(MODELS_DIR / "xboost_model.pkl")

# ── Build ring-score lookup at startup (fast dict lookup at request time) ─────
print("[startup] Building ring-score lookup from val + test data...")
_val  = pd.read_parquet(ROOT / "splitted_DS" / "val.parquet")
_test = pd.read_parquet(ROOT / "splitted_DS" / "test.parquet")
_all  = pd.concat([_val, _test], ignore_index=True)
_ring_df = detect_rings(_all)                                   # DataFrame: account_id, ring_score, pattern, detail
RING_LOOKUP: dict[str, dict] = {
    row["account_id"]: {
        "ring_score": row["ring_score"],
        "pattern":    row["pattern"],
        "detail":     row["detail"],
        "ring_chain": row["ring_chain"],
    }
    for _, row in _ring_df.iterrows()
}
del _val, _test, _all, _ring_df
print(f"[startup] Ring lookup ready — {len(RING_LOOKUP):,} accounts indexed.\n")

# ── Groq LLM ──────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="groq/compound",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    request_timeout=20,
)

# ── Audit log (append-only, one JSON line per call) ───────────────────────────
AUDIT_LOG_PATH = ROOT / "audit_log.jsonl"
_audit_lock = threading.Lock()

def _append_audit(record: dict) -> None:
    """Thread-safe, non-blocking append to audit_log.jsonl."""
    line = json.dumps(record, ensure_ascii=False)
    with _audit_lock:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# ── Constants ─────────────────────────────────────────────────────────────────
RING_WEIGHT = 0.40

FEATURES = [
    "amount", "log_amount", "recency_hours", "txn_count_24h", "is_dest_new", "hours_day",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
]


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class Transaction(BaseModel):
    # Account IDs (optional — used only for ring lookup; not needed by model)
    nameOrig: str = ""
    nameDest: str = ""
    # Model features
    amount: float
    recency_hours: float
    txn_count_24h: int
    is_dest_new: int
    hours_day: int
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type_CASH_IN: int
    type_CASH_OUT: int
    type_DEBIT: int
    type_PAYMENT: int
    type_TRANSFER: int


class PredictionResponse(BaseModel):
    # Model score
    fraud_probability: float
    is_fraud: bool
    decision: str
    risk_level: str
    # Ring scores
    sender_ring_score: float
    receiver_ring_score: float
    ring_score: float
    ring_pattern: str
    ring_detail: str
    in_ring: bool
    ring_chain: list[str]
    # Combined
    final_risk_score: float
    final_risk_level: str
    # Explanation
    summary: str

class ChatRequest(BaseModel):
    message: str
    context: dict


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: Transaction):

    # ── 1. Model score ────────────────────────────────────────────────────────
    txn_dict = transaction.dict()
    txn_dict["log_amount"] = np.log1p(txn_dict["amount"])

    df = pd.DataFrame([txn_dict])
    X  = df[FEATURES]

    fraud_prob = float(model.predict_proba(X)[0, 1])
    is_fraud   = fraud_prob >= 0.8

    if fraud_prob >= 0.8:
        risk_level = "HIGH"
    elif fraud_prob >= 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # ── 2. Ring scores ────────────────────────────────────────────────────────
    sender_info   = RING_LOOKUP.get(txn_dict["nameOrig"], {})
    receiver_info = RING_LOOKUP.get(txn_dict["nameDest"],  {})

    sender_ring_score   = sender_info.get("ring_score", 0.0)
    receiver_ring_score = receiver_info.get("ring_score", 0.0)
    ring_score          = max(sender_ring_score, receiver_ring_score)

    # Determine which party is the ring member (for display)
    if sender_ring_score >= receiver_ring_score and sender_ring_score > 0:
        ring_info = sender_info
    elif receiver_ring_score > 0:
        ring_info = receiver_info
    else:
        ring_info = {}

    ring_pattern = ring_info.get("pattern", "NONE")
    ring_detail  = ring_info.get("detail",  "No ring pattern detected")
    ring_chain   = [str(x) for x in ring_info.get("ring_chain", [])]  # always plain list[str]
    in_ring      = ring_score > 0

    # ── 3. Combined final_risk_score ──────────────────────────────────────────
    final_risk_score = float(
        np.clip(fraud_prob + (1 - fraud_prob) * ring_score * RING_WEIGHT, 0.0, 1.0)
    )

    if final_risk_score >= 0.8:
        final_risk_level = "HIGH"
    elif final_risk_score >= 0.5:
        final_risk_level = "MEDIUM"
    else:
        final_risk_level = "LOW"

    # ── 4. Determine transaction type label ───────────────────────────────────
    if txn_dict["type_TRANSFER"] == 1:
        txn_type = "TRANSFER"
    elif txn_dict["type_CASH_OUT"] == 1:
        txn_type = "CASH_OUT"
    elif txn_dict["type_PAYMENT"] == 1:
        txn_type = "PAYMENT"
    elif txn_dict["type_CASH_IN"] == 1:
        txn_type = "CASH_IN"
    else:
        txn_type = "DEBIT"

    # ── 5. Groq explanation ───────────────────────────────────────────────────
    ring_context = (
        f"- Ring Pattern: {ring_pattern} ({ring_detail})\n"
        f"- Ring Score: {ring_score:.4f}"
        if in_ring else
        "- No ring or layering pattern detected for this account"
    )

    prompt = f"""You are a fraud detection analyst. Analyze this transaction and provide a clear, professional summary.

ANALYSIS RESULTS:
- Fraud Probability (model): {fraud_prob*100:.2f}%
- Ring Score (graph analysis): {ring_score:.4f}
- Final Combined Risk Score: {final_risk_score*100:.2f}%
- Decision: {"FRAUD DETECTED" if is_fraud else "LEGITIMATE TRANSACTION"}
- Risk Level: {final_risk_level}

TRANSACTION DETAILS:
- Amount: ${txn_dict['amount']:,.2f}
- Type: {txn_type}
- Account Balance Before: ${txn_dict['oldbalanceOrg']:,.2f}
- Account Balance After: ${txn_dict['newbalanceOrig']:,.2f}
- Destination Balance Before: ${txn_dict['oldbalanceDest']:,.2f}
- Destination Balance After: ${txn_dict['newbalanceDest']:,.2f}
- New Destination: {"Yes" if txn_dict['is_dest_new'] == 1 else "No"}
- Hours Since Last Transaction: {txn_dict['recency_hours']}
- Transactions in Last 24h: {txn_dict['txn_count_24h']}

NETWORK ANALYSIS:
{ring_context}

Provide a 2-3 sentence professional summary explaining why this transaction is {"flagged as fraud" if is_fraud else "considered legitimate"}. Mention both the model score and any ring/layering patterns if present. Write in plain text only — no asterisks, no bold, no markdown, no bullet points."""

    try:
        response = llm.invoke([
            {"role": "system", "content": "You are a financial fraud detection expert. Be concise."},
            {"role": "user",   "content": prompt},
        ])
        summary = response.content.strip()
        # Strip any markdown the model sneaks in
        summary = re.sub(r'\*{1,2}(.+?)\*{1,2}', r'\1', summary)
        summary = re.sub(r'`(.+?)`', r'\1', summary)
    except Exception as e:
        summary = f"Error generating summary: {str(e)}"

    # ── 6. Audit log (fire-and-forget thread) ─────────────────────────────────
    audit_record = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "nameOrig":           txn_dict["nameOrig"],
        "nameDest":           txn_dict["nameDest"],
        "amount":             txn_dict["amount"],
        "txn_type":           txn_type,
        "fraud_probability":  round(fraud_prob, 6),
        "sender_ring_score":  round(sender_ring_score, 6),
        "receiver_ring_score": round(receiver_ring_score, 6),
        "ring_score":         round(ring_score, 6),
        "ring_pattern":       ring_pattern,
        "final_risk_score":   round(final_risk_score, 6),
        "final_risk_level":   final_risk_level,
        "decision":           "FRAUD" if is_fraud else "LEGITIMATE",
    }
    threading.Thread(target=_append_audit, args=(audit_record,), daemon=True).start()

    # ── 7. Return response ────────────────────────────────────────────────────
    return {
        "fraud_probability":    round(fraud_prob, 4),
        "is_fraud":             is_fraud,
        "decision":             "FRAUD" if is_fraud else "LEGITIMATE",
        "risk_level":           risk_level,
        "sender_ring_score":    round(sender_ring_score, 4),
        "receiver_ring_score":  round(receiver_ring_score, 4),
        "ring_score":           round(ring_score, 4),
        "ring_pattern":         ring_pattern,
        "ring_detail":          ring_detail,
        "ring_chain":           ring_chain,
        "in_ring":              in_ring,
        "final_risk_score":     round(final_risk_score, 4),
        "final_risk_level":     final_risk_level,
        "summary":              summary,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    ctx = req.context
    balance_drain = ctx.get('oldbalanceOrg', 0) > 0 and ctx.get('newbalanceOrig', 0) == 0
    dest_passthru  = ctx.get('oldbalanceDest', 0) == 0 and ctx.get('newbalanceDest', 0) == 0
    context_str = (
        f"=== TRANSACTION BEING ANALYSED ===\n"
        f"Type: {ctx.get('txn_type','?')}  |  Amount: ${ctx.get('amount',0):,.2f}\n"
        f"Hour of day: {ctx.get('hours_day','?')}  |  Hours since last txn: {ctx.get('recency_hours','?')}  |  Txns in last 24h: {ctx.get('txn_count_24h','?')}\n"
        f"New destination account: {'Yes' if ctx.get('is_dest_new') else 'No'}\n"
        f"\n=== BALANCE ANALYSIS ===\n"
        f"Sender balance before: ${ctx.get('oldbalanceOrg',0):,.2f}  ->  after: ${ctx.get('newbalanceOrig',0):,.2f}  (FULLY DRAINED: {balance_drain})\n"
        f"Receiver balance before: ${ctx.get('oldbalanceDest',0):,.2f}  ->  after: ${ctx.get('newbalanceDest',0):,.2f}  (PASS-THROUGH / zero balance: {dest_passthru})\n"
        f"\n=== SCORES ===\n"
        f"Fraud Score (model): {ctx.get('fraud_probability',0)*100:.1f}%\n"
        f"Sender ring score: {ctx.get('sender_ring_score',0)*100:.1f}%  |  Receiver ring score: {ctx.get('receiver_ring_score',0)*100:.1f}%\n"
        f"Ring Score: {ctx.get('ring_score',0)*100:.1f}%  |  Ring Pattern: {ctx.get('ring_pattern','NONE')}\n"
        f"Ring Detail: {ctx.get('ring_detail','No ring pattern detected')}\n"
        f"Ring Chain: {ctx.get('ring_chain', [])}\n"
        f"Final Risk Score: {ctx.get('final_risk_score',0)*100:.1f}%  |  Decision: {ctx.get('decision','?')}\n"
        f"\n=== KEY FRAUD SIGNALS TO USE IN YOUR ANSWER ===\n"
        f"- If balance_drain is True: the sender account was completely emptied — a classic fraud pattern\n"
        f"- If dest_passthru is True: money arrived at destination but immediately disappeared — relay/pass-through account\n"
        f"- If is_dest_new is Yes: first-time recipient, no prior relationship\n"
        f"- If recency_hours is very large: account was dormant then suddenly made a large transfer\n"
        f"- Ring pattern NONE means no graph ring was detected — the fraud score alone is driving the decision"
    )

    system = """You are a fraud detection assistant embedded in the SecurePay risk analysis tool.
You ONLY answer questions about:
- The specific transaction provided in the context
- Fraud detection concepts: ring scores, layering chains, fan-in hubs, model scoring, final risk
- How to interpret the scores shown

STRICT RULES:
- If the question is unrelated to this transaction or financial fraud, respond ONLY with: "I can only help with questions about this transaction or fraud detection."
- Never discuss coding, weather, news, general topics, or anything outside fraud/finance.
- Be concise — 2 to 4 sentences maximum.
- Do not use filler phrases like 'Great question' or 'Certainly'.
- Write in plain text only. No asterisks, no bold, no markdown formatting of any kind."""

    try:
        response = llm.invoke([
            {"role": "system", "content": system + "\n\n" + context_str},
            {"role": "user",   "content": req.message},
        ])
        text = response.content.strip()
        text = re.sub(r'\*{1,2}(.+?)\*{1,2}', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        return {"response": text}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import joblib
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
from openai import OpenAI
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

from pathlib import Path

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

MODELS_DIR = Path(__file__).resolve().parent
model = joblib.load(MODELS_DIR / "xboost_model.pkl")

# # client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY") # Ensure you have this environment variable set
)

features = [
    "amount", "log_amount", "recency_hours", "txn_count_24h", "is_dest_new", "hours_day",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"
]

class Transaction(BaseModel):
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
    fraud_probability: float
    is_fraud: bool
    decision: str
    risk_level: str
    summary: str

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: Transaction):

    txn_dict = transaction.dict()
    txn_dict['log_amount'] = np.log1p(txn_dict['amount'])
    
    df = pd.DataFrame([txn_dict])
    X = df[features]
    

    fraud_prob = float(model.predict_proba(X)[0, 1])
    is_fraud = fraud_prob >= 0.8
    

    if fraud_prob >= 0.8:
        risk_level = "HIGH"
    elif fraud_prob >= 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    

    if txn_dict['type_TRANSFER'] == 1:
        txn_type = "TRANSFER"
    elif txn_dict['type_CASH_OUT'] == 1:
        txn_type = "CASH_OUT"
    elif txn_dict['type_PAYMENT'] == 1:
        txn_type = "PAYMENT"
    elif txn_dict['type_CASH_IN'] == 1:
        txn_type = "CASH_IN"
    else:
        txn_type = "DEBIT"
    

    prompt = f"""You are a fraud detection analyst. Analyze this transaction and provide a clear, professional summary.

ANALYSIS RESULTS:
- Fraud Probability: {fraud_prob*100:.2f}%
- Decision: {"FRAUD DETECTED" if is_fraud else "LEGITIMATE TRANSACTION"}
- Risk Level: {risk_level}

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

Provide a 2-3 sentence professional summary explaining why this transaction is {"flagged as fraud" if is_fraud else "considered legitimate"}. Focus on the key risk indicators."""

    try:
        response = llm.invoke(
            [
                {"role": "system", "content": "You are a financial fraud detection expert."},
                {"role": "user", "content": prompt}
            ]
        )
        summary = response.content.strip()

    except Exception as e:
        summary = f"Error generating summary: {str(e)}"

    
    return {
        "fraud_probability": round(fraud_prob, 4),
        "is_fraud": is_fraud,
        "decision": "FRAUD" if is_fraud else "LEGITIMATE",
        "risk_level": risk_level,
        "summary": summary
    }
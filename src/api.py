"""
src/api.py
FastAPI app that loads the trained pipeline and serves predictions.
Run: uvicorn src.api:app --reload   (from project root)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import os

# ── Load model once at startup ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/pipeline.joblib")

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts whether a telecom customer will churn.",
    version="1.0.0",
)

pipeline = None

@app.on_event("startup")
async def load_model():
    global pipeline

    try:
        pipeline = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# ── Request / Response schemas ─────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    gender: str = Field("Male", example="Male")
    SeniorCitizen: int = Field(0, example=0)
    Partner: str = Field("Yes", example="Yes")
    Dependents: str = Field("No", example="No")
    tenure: int = Field(12, example=12)
    PhoneService: str = Field("Yes", example="Yes")
    MultipleLines: str = Field("No", example="No")
    InternetService: str = Field("Fiber optic", example="Fiber optic")
    OnlineSecurity: str = Field("No", example="No")
    OnlineBackup: str = Field("No", example="No")
    DeviceProtection: str = Field("No", example="No")
    TechSupport: str = Field("No", example="No")
    StreamingTV: str = Field("No", example="No")
    StreamingMovies: str = Field("No", example="No")
    Contract: str = Field("Month-to-month", example="Month-to-month")
    PaperlessBilling: str = Field("Yes", example="Yes")
    PaymentMethod: str = Field("Electronic check", example="Electronic check")
    MonthlyCharges: float = Field(70.35, example=70.35)
    TotalCharges: float = Field(844.2, example=844.2)


class PredictionResponse(BaseModel):
    churn_prediction: int          # 0 = No Churn, 1 = Churn
    churn_probability: float       # probability of churn
    risk_level: str                # Low / Medium / High


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert to DataFrame (preserves column names for the pipeline)
    data = pd.DataFrame([customer.model_dump()])

    pred = int(pipeline.predict(data)[0])
    prob = float(pipeline.predict_proba(data)[0][1])

    if prob < 0.35:
        risk = "Low"
    elif prob < 0.65:
        risk = "Medium"
    else:
        risk = "High"

    return PredictionResponse(
        churn_prediction=pred,
        churn_probability=round(prob, 4),
        risk_level=risk,
    )


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
    }

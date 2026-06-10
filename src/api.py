"""
src/api.py
FastAPI app that loads the trained pipeline and serves predictions.
Run: uvicorn src.api:app --reload (from project root)
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

# ── Model path (CI-safe) ───────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pipeline.joblib")

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts whether a telecom customer will churn.",
    version="1.0.0",
)

# ── Global model variable ─────────────────────────────────────────────────────
pipeline = None


# ── Lazy loader (FIX for CI + pytest + FastAPI lifecycle issues) ──────────────
def get_model():
    global pipeline

    if pipeline is None:
        try:
            pipeline = joblib.load(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    return pipeline


# ── Request schema ─────────────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int

    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    MonthlyCharges: float
    TotalCharges: float


# ── Response schema ────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None
    }


# ── Prediction endpoint ────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):

    # Convert request → DataFrame
    data = pd.DataFrame([customer.dict()])  # works with pydantic v1

    # Load model safely
    model = get_model()

    # Predict
    pred = int(model.predict(data)[0])
    prob = float(model.predict_proba(data)[0][1])

    # Risk classification
    if prob < 0.35:
        risk = "Low"
    elif prob < 0.65:
        risk = "Medium"
    else:
        risk = "High"

    return PredictionResponse(
        churn_prediction=pred,
        churn_probability=prob,
        risk_level=risk
    )


# ── Root endpoint ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Churn Prediction API",
        "docs": "/docs",
        "health": "/health"
    }

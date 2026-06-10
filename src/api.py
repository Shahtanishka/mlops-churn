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

# MLOps Churn Predictor

End-to-end MLOps pipeline predicting telecom customer churn — from raw data to a live REST API with experiment tracking, CI/CD, and drift monitoring.

## Architecture

```
Data → Preprocessing → Training (MLflow) → Model Registry
                                                  ↓
                                          FastAPI /predict
                                                  ↓
                                     Evidently drift monitor
                                                  ↓
                                       Retrain if drift detected
```

## Tech Stack

| Layer | Tool |
|---|---|
| Data | Pandas, scikit-learn Pipeline |
| Experiment tracking | MLflow |
| Model | XGBoost |
| Serving | FastAPI + Uvicorn |
| Monitoring | Evidently AI |
| CI/CD | GitHub Actions |

## Quickstart

```bash
# 1. Clone and set up
git clone https://github.com/YOUR_USERNAME/mlops-churn
cd mlops-churn
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 2. Train the model
python src/train.py

# 3. View MLflow experiments
mlflow ui   # → http://127.0.0.1:5000

# 4. Serve the API
uvicorn src.api:app --reload   # → http://127.0.0.1:8000/docs

# 5. Run drift monitoring
python src/monitor.py   # → reports/drift_report.html

# 6. Run tests
pytest tests/ -v
```

## API Usage

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 110.0, "TotalCharges": 220.0
  }'
```

**Response:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.7823,
  "risk_level": "High"
}
```

## Results

| Metric | Score |
|---|---|
| Accuracy | ~0.81 |
| ROC-AUC | ~0.85 |
| F1 (churn class) | ~0.62 |

*(Results vary slightly with synthetic data; use the real Kaggle dataset for full scores)*

## Real Dataset

Download the Telco Customer Churn dataset from Kaggle:
```
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
```
Save as `data/telco_churn.csv` and re-run `python src/train.py`.

## Project Structure

```
mlops-churn/
├── src/
│   ├── preprocess.py    # Data loading + sklearn pipeline
│   ├── train.py         # MLflow training script
│   ├── api.py           # FastAPI prediction server
│   └── monitor.py       # Evidently drift report
├── tests/
│   └── test_api.py      # Pytest API tests
├── models/              # Saved pipeline.joblib
├── reports/             # HTML drift reports
├── data/                # Dataset (gitignored)
├── .github/workflows/
│   └── ci.yml           # GitHub Actions CI
└── requirements.txt
```

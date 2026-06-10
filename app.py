"""
src/app.py
Streamlit UI for the Churn Prediction API.

Make sure the FastAPI server is running first:
    uvicorn src.api:app --reload

Then run this in a new terminal:
    streamlit run src/app.py
"""

import streamlit as st
import requests
import json

API_URL = "http://127.0.0.1:8000/predict"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #1A5276;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px 40px;
        width: 100%;
        border: none;
    }
    .stButton>button:hover { background-color: #154360; }
    .result-box {
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin-top: 16px;
    }
    .high   { background-color: #FADBD8; border: 2px solid #E74C3C; }
    .medium { background-color: #FDEBD0; border: 2px solid #E67E22; }
    .low    { background-color: #D5F5E3; border: 2px solid #27AE60; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 📡 Telecom Churn Predictor")
st.markdown("Fill in the customer details below and click **Predict** to see if they're likely to churn.")
st.markdown("---")

# ── Form layout: 3 columns ─────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 👤 Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    st.markdown("#### 📦 Services")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multi = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col3:
    st.markdown("#### 💳 Account")
    tenure = st.slider("Tenure (months)", min_value=1, max_value=72, value=12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    monthly = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=0.5)
    total = st.number_input("Total Charges ($)", min_value=18.0, max_value=8500.0, value=float(tenure * monthly))

st.markdown("---")

# ── Predict button ─────────────────────────────────────────────────────────────
predict_col, _ = st.columns([1, 2])
with predict_col:
    predict_clicked = st.button("🔍 Predict Churn")

# ── Prediction logic ───────────────────────────────────────────────────────────
if predict_clicked:
    payload = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": billing,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        result = response.json()

        pred  = result["churn_prediction"]
        prob  = result["churn_probability"]
        risk  = result["risk_level"]

        # ── Result display ─────────────────────────────────────────────────────
        st.markdown("### 🎯 Prediction Result")
        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color:#555">Prediction</h4>
                <h2 style="color:{'#E74C3C' if pred==1 else '#27AE60'}">
                    {'⚠️ Will Churn' if pred == 1 else '✅ Will Stay'}
                </h2>
            </div>""", unsafe_allow_html=True)

        with r2:
            bar_color = "#E74C3C" if prob > 0.65 else "#E67E22" if prob > 0.35 else "#27AE60"
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color:#555">Churn Probability</h4>
                <h2 style="color:{bar_color}">{prob*100:.1f}%</h2>
            </div>""", unsafe_allow_html=True)

        with r3:
            css_class = risk.lower()
            emoji = "🔴" if risk == "High" else "🟡" if risk == "Medium" else "🟢"
            st.markdown(f"""
            <div class="metric-card {css_class}">
                <h4 style="color:#555">Risk Level</h4>
                <h2>{emoji} {risk}</h2>
            </div>""", unsafe_allow_html=True)

        # ── Probability bar ────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Churn Probability**")
        st.progress(prob)

        # ── Insight ────────────────────────────────────────────────────────────
        st.markdown("#### 💡 Insight")
        if risk == "High":
            st.error(f"This customer has a **{prob*100:.1f}% chance of churning**. "
                     f"They are on a **{contract}** contract with **${monthly}/month** charges. "
                     f"Consider offering a discount or upgrading their plan.")
        elif risk == "Medium":
            st.warning(f"This customer has a **{prob*100:.1f}% chance of churning**. "
                       f"Monitor closely and consider a retention offer.")
        else:
            st.success(f"This customer has only a **{prob*100:.1f}% chance of churning**. "
                       f"They appear to be a loyal customer.")

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API. Make sure FastAPI is running:\n\n"
                 "```\nuvicorn src.api:app --reload\n```")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:13px'>"
    "MLOps Churn Predictor • Built with FastAPI + Streamlit + XGBoost + MLflow"
    "</p>",
    unsafe_allow_html=True
)

import streamlit as st
import joblib
import numpy as np
import os

# -----------------------------
# 1️⃣ Load Trained Pipeline
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "HeartDisease_model_pipeline.pkl")

try:
    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        st.error("❌ The loaded file is not a trained model. Please save the trained pipeline.")
        model = None
    else:
        st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model = None

# -----------------------------
# 2️⃣ User Inputs with Explanations
# -----------------------------
st.title("Heart Disease Prediction")

st.markdown("### Enter patient details:")

age = st.slider("Age", 0, 120, 50)
st.caption("Age of the patient in years.")

sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Male" if x==1 else "Female")
st.caption("Sex of the patient: Male or Female.")

chest_pain = st.selectbox(
    "Chest Pain Type", options=[0,1,2,3],
    format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",2:"Non-anginal Pain",3:"Asymptomatic"}[x]
)
st.caption("Type of chest pain experienced by the patient.")

rest_bp = st.slider("Resting Blood Pressure", 50, 250, 120)
st.caption("Patient's resting blood pressure in mmHg.")

cholesterol = st.slider("Serum Cholesterol", 100, 600, 200)
st.caption("Cholesterol level in mg/dl.")

fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
st.caption("Whether fasting blood sugar is greater than 120 mg/dl.")

ecg = st.selectbox(
    "Resting ECG Result", options=[0,1,2],
    format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",2:"Left Ventricular Hypertrophy"}[x]
)
st.caption("Result of the resting electrocardiogram test.")

max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150)
st.caption("Maximum heart rate achieved during exercise.")

angina = st.radio("Exercise Induced Angina", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
st.caption("Whether the patient has angina induced by exercise.")

st_dep = st.slider("ST Depression", 0.0, 10.0, 1.0, step=0.1)
st.caption("ST segment depression induced by exercise relative to rest.")

slope = st.selectbox(
    "Slope of ST Segment", options=[0,1,2],
    format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x]
)
st.caption("Slope of the peak exercise ST segment.")

vessels = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0,1,2,3])
st.caption("Number of major blood vessels colored by fluoroscopy (0-3).")

thal = st.selectbox(
    "Thalassemia", options=[1,2,3],
    format_func=lambda x: {1:"Normal",2:"Fixed Defect",3:"Reversible Defect"}[x]
)
st.caption("Thalassemia type: 1=Normal, 2=Fixed Defect, 3=Reversible Defect.")

# -----------------------------
# 3️⃣ Prepare Input Array
# -----------------------------
data = np.array([[age, sex, chest_pain, rest_bp, cholesterol, fbs, ecg, max_hr, angina, st_dep, slope, vessels, thal]])

# -----------------------------
# 4️⃣ Make Prediction with Probability
# -----------------------------
if st.button("Predict"):
    if model:
        try:
            probability = model.predict_proba(data)[0][1]  # probability of heart disease
            st.success(f"Prediction: {probability*100:.2f}% chance of heart disease")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
    else:
        st.error("Model is not loaded, cannot predict.")

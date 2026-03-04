import streamlit as st
import joblib
import numpy as np
import os

# -----------------------------
# 1️⃣ Load Trained Pipeline
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "heart_model.pkl")

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
# 2️⃣ User Inputs for 13 Features
# -----------------------------
st.title("Heart Disease Prediction")

# 1. Age
age = st.slider("Age", 0, 120, 50)

# 2. Sex
sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Male" if x==1 else "Female")

# 3. Chest Pain Type
chest_pain = st.selectbox("Chest Pain Type", options=[0,1,2,3],
                          format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",2:"Non-anginal Pain",3:"Asymptomatic"}[x])

# 4. Resting Blood Pressure
rest_bp = st.slider("Resting Blood Pressure", 50, 250, 120)

# 5. Serum Cholesterol
cholesterol = st.slider("Serum Cholesterol", 100, 600, 200)

# 6. Fasting Blood Sugar > 120 mg/dl
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")

# 7. Resting ECG Result
ecg = st.selectbox("Resting ECG Result", options=[0,1,2],
                   format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",2:"Left Ventricular Hypertrophy"}[x])

# 8. Max Heart Rate Achieved
max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150)

# 9. Exercise Induced Angina
angina = st.radio("Exercise Induced Angina", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")

# 10. ST Depression
st_dep = st.slider("ST Depression", 0.0, 10.0, 1.0, step=0.1)

# 11. Slope of ST Segment
slope = st.selectbox("Slope of ST Segment", options=[0,1,2],
                     format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])

# 12. Number of Major Vessels
vessels = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0,1,2,3])

# 13. Thalassemia
thal = st.selectbox("Thalassemia", options=[1,2,3],
                    format_func=lambda x: {1:"Normal",2:"Fixed Defect",3:"Reversible Defect"}[x])

# -----------------------------
# 3️⃣ Prepare Input Array
# -----------------------------
data = np.array([[age, sex, chest_pain, rest_bp, cholesterol, fbs, ecg, max_hr, angina, st_dep, slope, vessels, thal]])

# -----------------------------
# 4️⃣ Make Prediction
# -----------------------------
if st.button("Predict"):
    if model:
        try:
            prediction = model.predict(data)
            st.success(f"Prediction (0 = No Heart Disease, 1 = Heart Disease): {prediction[0]}")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
    else:
        st.error("Model is not loaded, cannot predict.")

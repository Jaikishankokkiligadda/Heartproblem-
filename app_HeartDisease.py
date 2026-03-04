import streamlit as st
import pickle
import numpy as np
import os

# -----------------------------
# 1️⃣ Set Model Path
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "heart_model.pkl")

# -----------------------------
# 2️⃣ Load Model Safely
# -----------------------------
model = None
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at: {model_path}")
else:
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")

# -----------------------------
# 3️⃣ User Input with Improved Widgets
# -----------------------------
st.title("Heart Disease Prediction")

# 1. Age (Slider)
age = st.slider("Age", 0, 120, 50)  # min 0, max 120, default 50

# 2. Sex (Radio)
sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Male" if x==1 else "Female"))

# 3. Chest Pain Type (Selectbox)
chest_pain = st.selectbox("Chest Pain Type", options=[0,1,2,3], 
                          format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",2:"Non-anginal Pain",3:"Asymptomatic"}[x])

# 4. Resting Blood Pressure (Slider)
rest_bp = st.slider("Resting Blood Pressure", 50, 250, 120)

# 5. Serum Cholesterol (Slider)
cholesterol = st.slider("Serum Cholesterol", 100, 600, 200)

# 6. Fasting Blood Sugar > 120 mg/dl (Radio)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")

# 7. Resting ECG Result (Selectbox)
ecg = st.selectbox("Resting ECG Result", options=[0,1,2], 
                   format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",2:"Left Ventricular Hypertrophy"}[x])

# 8. Max Heart Rate Achieved (Slider)
max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150)

# 9. Exercise Induced Angina (Radio)
angina = st.radio("Exercise Induced Angina", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")

# 10. ST Depression (Slider)
st_dep = st.slider("ST Depression", 0.0, 10.0, 1.0, step=0.1)

# 11. Slope of ST Segment (Selectbox)
slope = st.selectbox("Slope of ST Segment", options=[0,1,2], 
                     format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])

# 12. Number of Major Vessels (Selectbox)
vessels = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0,1,2,3])

# 13. Thalassemia (Selectbox)
thal = st.selectbox("Thalassemia", options=[1,2,3], 
                    format_func=lambda x: {1:"Normal",2:"Fixed Defect",3:"Reversible Defect"}[x])

# Collect all inputs in correct order
data = np.array([[age, sex, chest_pain, rest_bp, cholesterol, fbs, ecg, max_hr, angina, st_dep, slope, vessels, thal]])

# -----------------------------
# 4️⃣ Make Prediction Safely
# -----------------------------
if st.button("Predict"):
    if model is None:
        st.error("Model is not loaded, cannot predict.")
    else:
        try:
            prediction = model.predict(data)
            st.success(f"Prediction: {prediction[0]}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

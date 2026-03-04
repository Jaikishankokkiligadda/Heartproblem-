import streamlit as st
import pickle
import numpy as np
import os

# -----------------------------
# 1️⃣ Set Model Path
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "heart_model.pkl")  # ✅ updated file name

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
# 3️⃣ User Input with Real Feature Names
# -----------------------------
st.title("Heart Disease Prediction")

# Replace with your model's actual feature names
feature_names = [
    "Age",
    "Sex (1=Male, 0=Female)",
    "Chest Pain Type (0-3)",
    "Resting Blood Pressure",
    "Serum Cholesterol",
    "Fasting Blood Sugar > 120 mg/dl (1=Yes,0=No)",
    "Resting ECG Result (0-2)",
    "Max Heart Rate Achieved",
    "Exercise Induced Angina (1=Yes,0=No)",
    "ST Depression",
    "Slope of ST segment (0-2)",
    "Number of Major Vessels Colored by Fluoroscopy (0-3)",
    "Thalassemia (1-3)"
]

inputs = []
for feature in feature_names:
    val = st.number_input(feature, value=0.0)
    inputs.append(val)

# Convert to 2D array for prediction
data = np.array([inputs])

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

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
# 3️⃣ User Input (example)
# -----------------------------
st.title("Heart Disease Prediction")

# Example: assuming your model expects 13 features
inputs = []
for i in range(1, 14):
    val = st.number_input(f"Feature {i}", value=0.0)
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

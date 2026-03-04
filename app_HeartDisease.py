import streamlit as st
import pickle
import os
import numpy as np

# IMPORTANT: Import sklearn classes before loading
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier  # if you used xgboost

st.title("Heart Disease Prediction")

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "heart_model.pkl")  # <-- must match EXACT file name

if not os.path.exists(model_path):
    st.error("Model file not found!")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)

age = st.number_input("Age", 1, 120)

if st.button("Predict"):
    data = np.array([[age]])
    prediction = model.predict(data)
    st.success(f"Prediction: {prediction[0]}")

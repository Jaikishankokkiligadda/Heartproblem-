import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# 1️⃣ Load trained pipeline
# -----------------------------
model_path = "heart_model.pkl"

try:
    model = joblib.load(model_path)
    if not hasattr(model, "predict_proba"):
        st.error("❌ The loaded file is not a trained pipeline with predict_proba.")
        model = None
    else:
        st.success("Welcome to Check Your Heart!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model = None

st.title("Heart Disease Prediction App")

# -----------------------------
# 2️⃣ Option: Single Input or Batch CSV
# -----------------------------
option = st.radio("Select Input Type:", ["Single Patient Input", "Batch CSV Upload"])

if option == "Single Patient Input":
    st.subheader("Enter Patient Details:")

    age = st.slider("Age", 0, 120, 50)
    sex = st.radio("Sex", [0,1], format_func=lambda x: "Male" if x==1 else "Female")
    chest_pain = st.selectbox("Chest Pain Type", [0,1,2,3],
                              format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",
                                                     2:"Non-anginal Pain",3:"Asymptomatic"}[x])
    rest_bp = st.slider("Resting Blood Pressure", 50, 250, 120)
    cholesterol = st.slider("Serum Cholesterol", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    ecg = st.selectbox("Resting ECG Result", [0,1,2],
                       format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",2:"Left Ventricular Hypertrophy"}[x])
    max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    angina = st.radio("Exercise Induced Angina", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    st_dep = st.slider("ST Depression", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", [0,1,2],
                         format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])
    vessels = st.selectbox("Number of Major Vessels", [0,1,2,3])
    thal = st.selectbox("Thalassemia", [1,2,3],
                        format_func=lambda x: {1:"Normal",2:"Fixed Defect",3:"Reversible Defect"}[x])

    data = np.array([[age, sex, chest_pain, rest_bp, cholesterol, fbs, ecg,
                      max_hr, angina, st_dep, slope, vessels, thal]])

    if st.button("Predict Single Patient"):
        if model:
            probability = model.predict_proba(data)[0][1] * 100
            st.success(f"Prediction: {probability:.2f}% chance of heart disease")
        else:
            st.error("Model is not loaded, cannot predict.")

else:
    st.subheader("Upload CSV File for Batch Prediction")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            ids = batch_df['id'] if 'id' in batch_df.columns else pd.Series(range(len(batch_df)))
            X_batch = batch_df.drop(columns=['id'], errors='ignore')

            predictions = model.predict_proba(X_batch)[:,1] * 100
            results = pd.DataFrame({
                "id": ids,
                "Heart_Disease_Probability(%)": np.round(predictions,2)
            })

            st.subheader("Batch Prediction Results")
            st.dataframe(results)

            # Download button
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name="heart_batch_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

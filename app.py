import streamlit as st
import numpy as np
import joblib
import pandas as pd

model = joblib.load("diabetes_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="wide")

st.title("🩺 Diabetes Prediction System")
st.markdown("This application uses a Support Vector Machine (SVM) model to predict diabetes risk.")

st.sidebar.header("Patient Information")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.number_input("Glucose", 0, 300, 120)
blood_pressure = st.sidebar.number_input("Blood Pressure", 0, 200, 70)
skin_thickness = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.number_input("Age", 1, 120, 30)

if st.button("Predict Diabetes Risk"):
    data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                      insulin, bmi, dpf, age]])
    scaled = scaler.transform(data)
    prediction = model.predict(scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Risk: Patient is likely diabetic.")
    else:
        st.success("✅ Low Risk: Patient is likely not diabetic.")

st.caption("Developed with Streamlit, Scikit-Learn and SVM")

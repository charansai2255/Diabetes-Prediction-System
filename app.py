import streamlit as st
import numpy as np
import joblib

# ============================
# LOAD MODEL & SCALER
# ============================

model = joblib.load("diabetes_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# ============================
# PAGE CONFIG
# ============================

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ============================
# CUSTOM CSS
# ============================

st.markdown("""
<style>
.main {
    padding: 2rem;
}
.stButton > button {
    width: 100%;
    height: 50px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ============================
# HEADER
# ============================

st.title("🩺 Diabetes Prediction System")
st.markdown(
    """
    This application uses a Support Vector Machine (SVM) Machine Learning model
    to predict diabetes risk based on patient health information.
    """
)

# ============================
# SIDEBAR INPUTS
# ============================

st.sidebar.header("Patient Information")

pregnancies = st.sidebar.number_input(
    "Pregnancies", min_value=0, max_value=20, value=1
)

glucose = st.sidebar.number_input(
    "Glucose", min_value=0, max_value=300, value=120
)

blood_pressure = st.sidebar.number_input(
    "Blood Pressure", min_value=0, max_value=200, value=70
)

skin_thickness = st.sidebar.number_input(
    "Skin Thickness", min_value=0, max_value=100, value=20
)

insulin = st.sidebar.number_input(
    "Insulin", min_value=0, max_value=900, value=80
)

bmi = st.sidebar.number_input(
    "BMI", min_value=0.0, max_value=70.0, value=25.0
)

dpf = st.sidebar.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.sidebar.number_input(
    "Age", min_value=1, max_value=120, value=30
)

# ============================
# DASHBOARD METRICS
# ============================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Age", age)

with col2:
    st.metric("BMI", bmi)

with col3:
    st.metric("Glucose", glucose)

# ============================
# PREDICTION BUTTON
# ============================

if st.button("Predict Diabetes Risk"):

    # Prepare input
    data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    # Scale input
    scaled_data = scaler.transform(data)

    # Prediction
    prediction = model.predict(scaled_data)

    # Probability
    probability = model.predict_proba(scaled_data)

    risk_score = probability[0][1] * 100

    # ============================
    # RESULTS
    # ============================

    st.subheader("Prediction Results")

    st.metric(
        "Diabetes Risk Score",
        f"{risk_score:.2f}%"
    )

    if prediction[0] == 1:
        st.error("⚠️ High Risk: Patient is likely diabetic.")
    else:
        st.success("✅ Low Risk: Patient is likely not diabetic.")

    # ============================
    # BMI ANALYSIS
    # ============================

    st.subheader("BMI Analysis")

    if bmi < 18.5:
        st.info("Underweight")
    elif bmi < 25:
        st.success("Normal Weight")
    elif bmi < 30:
        st.warning("Overweight")
    else:
        st.error("Obese")

    # ============================
    # RISK FACTORS
    # ============================

    risk_factors = []

    if glucose > 140:
        risk_factors.append("High Glucose")

    if bmi > 30:
        risk_factors.append("High BMI")

    if age > 45:
        risk_factors.append("Age Above 45")

    if blood_pressure > 90:
        risk_factors.append("High Blood Pressure")

    if risk_factors:

        st.subheader("Risk Factors")

        for factor in risk_factors:
            st.warning(factor)

# ============================
# DISCLAIMER
# ============================

st.markdown("---")

st.warning(
    "This application is intended for educational purposes only. "
    "It should not be used as a substitute for professional medical advice, "
    "diagnosis, or treatment."
)

st.caption(
    "Developed using Python, Streamlit, Scikit-Learn, and Support Vector Machine (SVM)"
)
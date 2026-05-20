import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Clinical Sepsis Risk AI",
    page_icon="🩺",
    layout="centered"
)

# --- LOAD TRAINED MODEL ---
MODEL_PATH = "models/sepsis_random_forest.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

# --- APP INTERFACE ---
st.title("🩺 Clinical Sepsis Early-Warning Dashboard")
st.write("""
This enterprise-grade predictive system utilizes an optimized Random Forest classifier 
to assess patient systemic infection risk based on core physiological and laboratory biomarkers.
""")
st.markdown("---")

if model is None:
    st.error(f"❌ Model artifact not found at `{MODEL_PATH}`. Please verify that your notebook successfully saved the model file.")
else:
    st.subheader("📊 Patient Clinical Metrics Entry")
    st.write("Adjust the sliders below to match the patient's incoming ICU laboratory and vital stats:")

    col1, col2 = st.columns(2)

    with col1:
        lactate = st.slider("Lactate (mmol/L)", min_value=0.0, max_value=15.0, value=1.2, step=0.1,
                            help="Normal: < 2.0 mmol/L. High lactate signals tissue hypoxia.")
        creatinine = st.slider("Creatinine (mg/dL)", min_value=0.0, max_value=10.0, value=0.9, step=0.1,
                               help="Normal: 0.6 - 1.2 mg/dL. Elevated levels point to kidney stress.")
        sirs = st.slider("SIRS Criteria Score", min_value=0, max_value=4, value=1, step=1,
                         help="Systemic Inflammatory Response Syndrome score (>=2 is critical).")
        sofa = st.slider("SOFA Score", min_value=0, max_value=24, value=2, step=1,
                         help="Sequential Organ Failure Assessment score tracking organ dysfunction.")
        qsofa = st.slider("qSOFA Score", min_value=0, max_value=3, value=1, step=1,
                          help="Quick SOFA score for rapid bedside assessment.")

    with col2:
        apache = st.slider("APACHE IV Score", min_value=0, max_value=286, value=45, step=1,
                           help="Acute Physiology and Chronic Health Evaluation severity score.")
        resp_rate = st.slider("Respiratory Rate (Mean bpm)", min_value=8, max_value=50, value=18, step=1)
        hr_mean = st.slider("Heart Rate (Mean bpm)", min_value=40, max_value=200, value=82, step=1)
        age = st.slider("Patient Age (Years)", min_value=18, max_value=100, value=62, step=1)

    st.markdown("---")

    # --- INFERENCE ENGINE ---
    input_data = pd.DataFrame([[
        creatinine, lactate, sirs, apache, sofa, resp_rate, qsofa, hr_mean, age
    ]], columns=['creatinine', 'lactate_mmol', 'sirs_criteria', 'apache_iv', 'sofa_score', 'respiratory_rate_mean', 'qsofa', 'hr_mean', 'age'])

    # Calculate predictions
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    sepsis_risk = probabilities[1] * 100

    # --- DISPLAY RESULTS ---
    st.subheader("🚨 Machine Learning Risk Assessment")
    st.progress(int(sepsis_risk))
    
    if prediction == 1:
        st.error(f"🔴 **HIGH RISK DETECTION:** The model predicts a **{sepsis_risk:.2f}%** probability of Sepsis. Immediate clinical intervention recommended.")
    else:
        if sepsis_risk > 30:
            st.warning(f"🟡 **BORDERLINE ELEVATED RISK:** The model predicts a **{sepsis_risk:.2f}%** probability of Sepsis. Monitor patient biomarkers closely.")
        else:
            st.success(f"🟢 **LOW RISK:** The model predicts a **{sepsis_risk:.2f}%** probability of Sepsis. Patient metrics are within stable boundaries.")

    with st.expander("🛠️ View Pipeline Technical Metadata"):
        st.write("Features passed to `model.predict()` in real-time shape:")
        st.dataframe(input_data)
        st.code(f"Raw Model Probability Output: [Healthy: {probabilities[0]:.4f}, Sepsis: {probabilities[1]:.4f}]")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Healthcare Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for shadows, cards, and styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained KNN model
@st.cache_resource
def load_model():
    model_path = 'knn.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    return None

model = load_model()

# Header Section
st.markdown("""
    <div class="card" style="text-align: center; background: linear-gradient(135deg, #6e8efb, #a777e3); color: white;">
        <h1 style="margin: 0;">🏥 Healthcare Cost Prediction App</h1>
        <p style="font-size: 1.1em; margin-top: 5px;">Powered by K-Nearest Neighbors (KNN) Regressor</p>
    </div>
""", unsafe_allow_Html=True)

if model is None:
    st.error("⚠️ Model file `knn.pkl` not found in the root directory. Please upload it alongside `app.py`.")
else:
    with st.form("prediction_form"):
        st.markdown("### 📋 Enter Patient & Lifestyle Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("👤 Biometrics & Demographics")
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
            city_type = st.selectbox("City Type", ["Urban", "Semi-Urban", "Rural"])
            insurance_type = st.selectbox("Insurance Type", ["Private", "Government", "None"])
            insurance_coverage_pct = st.slider("Insurance Coverage (%)", 0.0, 100.0, 80.0)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🩺 Medical History")
            smoker = st.selectbox("Smoker", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
            asthma = st.selectbox("Asthma", ["No", "Yes"])
            doctor_visits_per_year = st.number_input("Doctor Visits / Year", min_value=0, max_value=50, value=2)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🏃‍♂️ Lifestyle & Utilization")
            physical_activity_level = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
            daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=7500)
            sleep_hours = st.number_input("Sleep Hours (Daily)", min_value=0.0, max_value=24.0, value=7.5)
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
            hospital_admissions = st.number_input("Hospital Admissions", min_value=0, max_value=20, value=0)
            medication_count = st.number_input("Medication Count", min_value=0, max_value=50, value=1)
            previous_year_cost = st.number_input("Previous Year Cost ($)", min_value=0.0, max_value=100000.0, value=1200.0)
            st.markdown('</div>', unsafe_allow_html=True)
            
        submitted = st.form_submit_button("Predict Estimated Cost")
        
        if submitted:
            # Map categorical variables to match your model's expected encoding
            input_data = pd.DataFrame({
                'age': [age],
                'bmi': [bmi],
                'smoker': [1 if smoker == "Yes" else 0],
                'diabetes': [1 if diabetes == "Yes" else 0],
                'hypertension': [1 if hypertension == "Yes" else 0],
                'heart_disease': [1 if heart_disease == "Yes" else 0],
                'asthma': [1 if asthma == "Yes" else 0],
                'physical_activity_level': [2 if physical_activity_level == "High" else (1 if physical_activity_level == "Moderate" else 0)],
                'daily_steps': [daily_steps],
                'sleep_hours': [sleep_hours],
                'stress_level': [stress_level],
                'doctor_visits_per_year': [doctor_visits_per_year],
                'hospital_admissions': [hospital_admissions],
                'medication_count': [medication_count],
                'insurance_type': [2 if insurance_type == "Private" else (1 if insurance_type == "Government" else 0)],
                'insurance_coverage_pct': [insurance_coverage_pct],
                'city_type': [1 if city_type == "Urban" else (0 if city_type == "Semi-Urban" else 2)],
                'previous_year_cost': [previous_year_cost]
            })
            
            try:
                prediction = model.predict(input_data)[0]
                st.markdown(f"""
                    <div class="card" style="text-align: center; background-color: #e6f4ea; border-left: 6px solid #34a853;">
                        <h2 style="color: #137333; margin: 0;">Predicted Medical Cost</h2>
                        <h1 style="color: #0d652d; margin-top: 10px;">${prediction:,.2f}</h1>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error during prediction: {e}")

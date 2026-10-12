import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained KNN model
@st.cache_resource
def load_model():
    with open('knn.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("Student Performance Prediction App")
st.write("Enter the details below to get a prediction from your KNN model[cite: 1].")

# Collect user inputs for each model feature[cite: 1]
age = st.number_input("Age", min_value=10, max_value=100, value=20)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
course = st.text_input("Course", "Computer Science")
study_hours = st.slider("Study Hours per Day", 0.0, 15.0, 4.0)
class_attendance = st.slider("Class Attendance (%)", 0.0, 100.0, 85.0)
internet_access = st.selectbox("Internet Access", ["Yes", "No"])
sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Average", "Good", "Excellent"])
study_method = st.selectbox("Study Method", ["Self-study", "Online Videos", "Group Study", "Classes"])
facility_rating = st.slider("Facility Rating (1-5)", 1, 5, 3)
exam_difficulty = st.slider("Exam Difficulty (1-5)", 1, 5, 3)

# Prediction button
if st.button("Predict"):
    # Pack input data into a DataFrame matching training features
    input_data = pd.DataFrame([[
        age, gender, course, study_hours, class_attendance, 
        internet_access, sleep_hours, sleep_quality, study_method, 
        facility_rating, exam_difficulty
    ]], columns=[
        'age', 'gender', 'course', 'study_hours', 'class_attendance', 
        'internet_access', 'sleep_hours', 'sleep_quality', 'study_method', 
        'facility_rating', 'exam_difficulty'
    ])
    
    try:
        # Note: If your model was trained on numerically encoded/scaled features, 
        # ensure you apply the exact same transformation here before calling predict().
        prediction = model.predict(input_data)
        st.success(f"Predicted Output: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Tip: If your model expects encoded numerical values or specific scaling, make sure to transform the inputs accordingly.")

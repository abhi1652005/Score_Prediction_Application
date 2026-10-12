import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for shadow layout and modern styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        color: white;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 12px;
        background: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained KNN model
@st.cache_resource
def load_model():
    with open('knn.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# App Header
st.title("🎓 Student Performance Prediction App")
st.markdown("Provide the student details below to predict performance outcomes using your trained **K-Nearest Neighbors** regression model[cite: 4].")
st.markdown("---")

# Structured input layout using a form and 3 columns
with st.form("prediction_form"):
    st.subheader("📋 Enter Student Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=20, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        course = st.selectbox("Course / Field of Study", ["Science", "Commerce", "Arts", "Engineering", "Other"])
        study_hours = st.slider("Study Hours (per day)", min_value=0.0, max_value=16.0, value=4.0, step=0.5)
        
    with col2:
        class_attendance = st.slider("Class Attendance (%)", min_value=0, max_value=100, value=85, step=1)
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        sleep_hours = st.slider("Sleep Hours (per night)", min_value=1.0, max_value=12.0, value=7.0, step=0.5)
        sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Average", "Good", "Excellent"])
        
    with col3:
        study_method = st.selectbox("Study Method", ["Self-Study", "Group Study", "Online Courses", "Coaching"])
        facility_rating = st.slider("Facility Rating (1-5)", min_value=1, max_value=5, value=3, step=1)
        exam_difficulty = st.selectbox("Exam Difficulty", ["Easy", "Medium", "Hard"])
    
    st.markdown("")
    submit_button = st.form_submit_button(label="🚀 Predict Performance")

# Handle prediction logic upon form submission
if submit_button:
    # NOTE: Adjust these mappings if your training pipeline used specific Label Encoders
    gender_val = 1 if gender == "Male" else (0 if gender == "Female" else 2)
    internet_val = 1 if internet_access == "Yes" else 0
    
    course_mapping = {"Science": 0, "Commerce": 1, "Arts": 2, "Engineering": 3, "Other": 4}
    sleep_quality_mapping = {"Poor": 1, "Average": 2, "Good": 3, "Excellent": 4}
    study_method_mapping = {"Self-Study": 0, "Group Study": 1, "Online Courses": 2, "Coaching": 3}
    exam_difficulty_mapping = {"Easy": 1, "Medium": 2, "Hard": 3}
    
    features = np.array([[
        age,
        gender_val,
        course_mapping.get(course, 0),
        study_hours,
        class_attendance,
        internet_val,
        sleep_hours,
        sleep_quality_mapping.get(sleep_quality, 2),
        study_method_mapping.get(study_method, 0),
        facility_rating,
        exam_difficulty_mapping.get(exam_difficulty, 2)
    ]])
    
    try:
        prediction = model.predict(features)
        
        # Display Result Card with Shadow Effect
        st.markdown(f"""
            <div class="prediction-card">
                <h3>Predicted Performance Score</h3>
                <h1 style="color: #4F46E5; font-size: 3rem;">{prediction[0]:.2f}</h1>
                <p>Generated successfully using your KNeighborsRegressor model[cite: 4].</p>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Tip: Ensure the input feature order and categorical encodings match the exact format your model was trained on.")

# Sidebar documentation
with st.sidebar:
    st.header("About")
    st.markdown("This app deploys a machine learning regression model to estimate student performance based on academic habits and lifestyle indicators[cite: 4].")
    st.markdown("---")
    st.markdown("### Required Features[cite: 4]:")
    st.markdown("""
    - Age[cite: 4]
    - Gender[cite: 4]
    - Course[cite: 4]
    - Study Hours[cite: 4]
    - Class Attendance[cite: 4]
    - Internet Access[cite: 4]
    - Sleep Hours[cite: 4]
    - Sleep Quality[cite: 4]
    - Study Method[cite: 4]
    - Facility Rating[cite: 4]
    - Exam Difficulty[cite: 4]
    """)

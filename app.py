import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib

# Set the background of the Streamlit web app to the American Diabetes Association Logo URL
background_image_url = "https://diabetes.org/sites/default/files/ADA85_logo_full_RGB_0.svg"


# Set and inject CSS code to change the background to the image URL above
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("{background_image_url}");
            background-size: contain;
            background-position: center center;
            background-repeat: no-repeat;
        }}
    </style>
    """, 
    unsafe_allow_html=True
)

# Load the pre-trained model and scaler
@st.cache_resource
def load_model():
    model = keras.models.load_model("diabetes_model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# Function to read training log
def read_training_log():
    try:
        with open("training_log.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "No training log available."

# Streamlit UI
st.title("Diabetes Risk Prediction Model")

# Display training progress log in a text box
st.text_area("Model Training Results:", read_training_log(), height=300)

# User Input Section
col1, col2 = st.columns([2, 1])
with col1:
    weight = st.number_input("What is your approximate weight? (lbs)", min_value=50.0, max_value=700.0, step=0.1)
    height_feet = st.number_input("What is your height? (feet)", min_value=3, max_value=8, step=1)
    height_inches = st.number_input("What is your height? (additional inches)", min_value=0, max_value=11, step=1)

# Calculate BMI
total_height_inches = (height_feet * 12) + height_inches
bmi = (weight / (total_height_inches ** 2)) * 703 if total_height_inches > 0 else 0
st.write(f"Your calculated Body Mass Index (BMI): **{bmi:.2f}**")

# Three-column layout for radio buttons
# Row 1: Age, General Health, Gender
col1, col2, col3 = st.columns(3)
with col1:
    selected_age = st.radio("Select your age group:", ["18-24 years", "25-29 years", "30-34 years", "35-39 years", 
                                                       "40-44 years", "45-49 years", "50-54 years", "55-59 years", 
                                                       "60-64 years", "65-69 years", "70-74 years", "75-79 years", 
                                                       "80+ years"])
with col2:
    selected_health = st.radio("Select your general health level:", ["Excellent", "Very Good", "Good", "Fair", "Poor"])
with col3:
    selected_sex = st.radio("Select your gender:", ["Male", "Female"])

# Row 2: Education, Income (third column left empty)
col1, col2 = st.columns(2)
with col1:
    selected_education = st.radio("Select your Education Level", ["Never attended school or only kindergarten", 
                                                                  "Grades 1-8 (Elementary)", 
                                                                  "Grades 9-11 (Some high school)", 
                                                                  "Grade 12 or GED (High school graduate)",
                                                                  "College 1-3 years (Some college/tech school)",
                                                                  "College Graduate"])
with col2:
    selected_income = st.radio(
        "Select your Income Level",
        [
            "Less than \$10,000 per year",
            "\$10,000 - \$15,000",
            "\$15,000 - \$25,000",
            "\$25,000 - \$35,000",
            "\$35,000 - \$50,000",  
            "\$50,000 - \$65,000",
            "\$65,000 - \$75,000",
            "More than \$75,000"
        ]
    )
# with col3:
#     pass  # Leave empty for now

# Additional numeric inputs
ment_hlth = st.number_input("How many days in the last 30 days would you say you had overall bad mental health?", min_value=0, max_value=30, step=1)
phys_hlth = st.number_input("How many days in the last 30 days would you say you had overall bad physical health?", min_value=0, max_value=30, step=1)

# Feature selection checkboxes
feature_mapping = {
    "Has high blood pressure (130/80 mmHg or higher)": "HighBP",
    "Has high cholesterol (240+ total cholesterol)": "HighChol",
    "Had your cholesterol checked in the last 5 years": "CholCheck",
    "Have smoked at least 100 cigarettes (5 packs) in your lifetime": "Smoker",
    "Have a history of stroke": "Stroke",
    "Have a history of Heart Disease or Heart Attack": "HeartDiseaseorAttack",
    "Had Physical Activity in the past 30 days not including work": "PhysActivity",
    "Eats at least 1 helping of fruits a day": "Fruits",
    "Eats at least 1 helping of vegetables a day": "Veggies",
    "Drinks more than 7 alcoholic beverages per week": "HvyAlcoholConsump",
    "Have healthcare coverage of any kind?": "AnyHealthcare",
    "Couldn't see a doctor in the last 12 months due to high cost": "NoDocbcCost",
    "Has difficulty walking or climbing stairs": "DiffWalk",
}

checkbox_values = {col: st.checkbox(label, value=False) for label, col in feature_mapping.items()}

# Mappings for categorical inputs
age_mapping = {
    "18-24 years": 1, "25-29 years": 2, "30-34 years": 3, "35-39 years": 4, "40-44 years": 5,
    "45-49 years": 6, "50-54 years": 7, "55-59 years": 8, "60-64 years": 9, "65-69 years": 10,
    "70-74 years": 11, "75-79 years": 12, "80+ years": 13
}
selected_age_int = age_mapping[selected_age]

health_mapping = {
    "Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5
}
selected_health_int = health_mapping[selected_health]

gender_mapping = {
    "Male": 1, "Female": 0
}
selected_sex_int = gender_mapping[selected_sex]

education_mapping = {
    "Never attended school or only kindergarten": 1,
    "Grades 1-8 (Elementary)": 2,
    "Grades 9-11 (Some high school)": 3,
    "Grade 12 or GED (High school graduate)": 4,
    "College 1-3 years (Some college/tech school)": 5,
    "College Graduate": 6
}
selected_education_int = education_mapping[selected_education]

income_mapping = {
    "Less than \$10,000 per year": 1,
    "\$10,000 - \$15,000": 2,
    "\$15,000 - \$25,000": 3,
    "\$25,000 - \$35,000": 4,
    "\$35,000 - \$50,000": 5,
    "\$50,000 - \$65,000": 6,
    "\$65,000 - \$75,000": 7,
    "More than \$75,000": 8
}
selected_income_int = income_mapping[selected_income]

# Prediction button
if st.button("Predict Risk of Diabetes"):
    user_input = {
        "BMI": int(round(bmi, 0)),
        "Age": int(selected_age_int),
        "GenHlth": int(selected_health_int),
        "Sex": int(selected_sex_int),
        "MentHlth": int(ment_hlth),
        "PhysHlth": int(phys_hlth),
        "Education": int(selected_education_int),
        "Income": int(selected_income_int),
        **{col: int(val) for col, val in checkbox_values.items()}
    }

    # Convert to DataFrame and ensure the input matches expected features, then scale
    input_df = pd.DataFrame([user_input])

    # Ensure input DataFrame matches the expected feature names and order
    expected_features = list(scaler.feature_names_in_)  # Get expected feature names in the correct order
    input_df = input_df.reindex(columns=expected_features, fill_value=0)  # Align columns

    # Transform input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = (model.predict(input_scaled) > 0.5)

    # Display result
    st.subheader("Predicted Diabetes Risk:")
    if prediction[0][0]:  # Adjusted for array indexing
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")


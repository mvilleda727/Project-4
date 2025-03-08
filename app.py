# Import depencies
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time  # Import time for better UI updates

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

# Load dataset from the CSV file in the Resources folder
file_path = "./Resources/diabetes_binary_health_indicators_BRFSS2015.csv"
df = pd.read_csv(file_path)

# Map human-readable names for each column in the dataset so the checkboxes the user can select are easy to read
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

# Define age categories with corresponding dataset values from the dataset
age_categories = {
    "18-24 years": 1,
    "25-29 years": 2,
    "30-34 years": 3,
    "35-39 years": 4,
    "40-44 years": 5,
    "45-49 years": 6,
    "50-54 years": 7,
    "55-59 years": 8,
    "60-64 years": 9,
    "65-69 years": 10,
    "70-74 years": 11,
    "75-79 years": 12,
    "80+ years": 13
}

# Define categories of sex/gender so the user can select
sex_choice = {
    "Male": 1,
    "Female": 0
}

# Streamlit UI
st.title("Diabetes Risk Prediction Model")
st.write("Enter and select factors that apply to you to predict your statistical risk of having diabetes")

# Create a dictionary to store user inputs
user_input = {}

# Create two columns: one for the input fields (BMI-related) and one for the sex radio button
col1, col2 = st.columns([2, 1])  # Make the first column wider (2) and the second column narrower (1)

# First column: user input fields (BMI-related)
with col1:
    weight = st.number_input("What is your approximate weight? (lbs)", min_value=50.0, max_value=700.0, step=0.1)
    height_feet = st.number_input("What is your height? (feet)", min_value=3, max_value=8, step=1)
    height_inches = st.number_input("What is your height? (additional inches)", min_value=0, max_value=11, step=1)

# Convert height to total inches
total_height_inches = (height_feet * 12) + height_inches

# Calculate BMI using the formula: BMI = (weight (lbs) / height (inches)^2) * 703
if total_height_inches > 0:
    bmi = (weight / (total_height_inches ** 2)) * 703
    st.write(f"Your calculated Body Mass Index (BMI): **{bmi:.2f}**")
else:
    bmi = 0  # Default value if height is not entered

st.write("")
st.write("")
st.write("")

user_input["BMI"] = bmi  # Store the calculated BMI

# Create three columns for the radio button lists (Age, General Health, and Sex)
col1, col2, col3 = st.columns(3)

# First column: Age radio button list
with col1:
    selected_age = st.radio("Select your age group:", list(age_categories.keys()))
    user_input["Age"] = age_categories[selected_age]  # Convert selected label to dataset value

# Second column: General Health radio button list
with col2:
    general_health_choices = {
        "Excellent": 1,
        "Very Good": 2,
        "Good": 3,
        "Fair": 4,
        "Poor": 5
    }
    selected_health = st.radio("Select your general health level:", list(general_health_choices.keys()))
    user_input["GenHlth"] = general_health_choices[selected_health]

# Add lines between objects for easier viewing
st.write("")
st.write("")
st.write("")

# Third column: Sex radio button list
with col3:
    selected_sex = st.radio("Select your gender:", list(sex_choice.keys()))
    user_input["Sex"] = sex_choice[selected_sex]

# New user input for mental health and physical health days (using number input fields like BMI)
user_input["MentHlth"] = st.number_input("How many days of bad mental health did you have in the past 30 days? (input range 1 to 30)", min_value=0, max_value=30, step=1)
user_input["PhysHlth"] = st.number_input("How many days of bad physical health did you have in the past 30 days? (input range 1 to 30)", min_value=0, max_value=30, step=1)

# Add lines between objects for easier viewing
st.write("")
st.write("")
st.write("")

# Divide the checkboxes into columns for better space distribution
num_columns = 3  # Number of columns you want for the checkboxes
columns = st.columns(num_columns)

# Generate checkboxes with friendly names but store actual column values (No "Education" or "Income")
checkbox_list = list(feature_mapping.items())

# Distribute checkboxes across columns
for i, (display_name, column_name) in enumerate(checkbox_list):
    col = columns[i % num_columns]  # Use modulo to distribute across columns
    user_input[column_name] = 1 if col.checkbox(display_name, value=False) else 0  # Default unchecked

# Convert user input to a DataFrame
input_df = pd.DataFrame([user_input])

# Add lines between objects for easier viewing
st.write("")
st.write("")
st.write("")

# Create a button the user can click on to begin the prediction model training
if st.button("Click here to begin training the prediction model"):
    # Placeholder for live logs
    log_placeholder = st.empty()

    # Normalize dataset features (but keep user input separate)
    X = df[list(feature_mapping.values()) + ["BMI", "Age", "Sex"]]  # Include BMI and Age, but not Education or Income
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Fit scaler on dataset features

    # Ensure input DataFrame matches training columns and convert to NumPy array
    expected_columns = list(feature_mapping.values()) + ["BMI", "Age", "Sex"]
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)  # Fix column mismatch issue
    input_scaled = scaler.transform(input_df.values)  # Convert to NumPy array before transforming

    # Separate target variable
    y = df["Diabetes_binary"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define model
    model = keras.Sequential([ 
        keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Custom callback to show real-time epoch updates
    class StreamlitCallback(keras.callbacks.Callback):
        def __init__(self, log_placeholder):
            super().__init__()
            self.log_placeholder = log_placeholder
            self.log_text = ""  # Store all log outputs

        def on_epoch_begin(self, epoch, logs=None):
            """Display the start of an epoch immediately."""
            self.log_text += f"Running Epoch {epoch+1}/50...\n"
            self.log_placeholder.text_area("Training Progress:", self.log_text, height=300)
            time.sleep(0.1)  # Small delay for better UI updates

        def on_epoch_end(self, epoch, logs=None):
            """Update loss and accuracy once epoch completes."""
            if logs:
                self.log_text += f"Epoch {epoch+1}/50 - Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}\n"
                self.log_placeholder.text_area("Training Progress:", self.log_text, height=300)
                time.sleep(0.1)  # Small delay to make updates smoother

    # Train the model with live updates
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=0, callbacks=[StreamlitCallback(log_placeholder)])

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.write(f"Model Test Accuracy: **{accuracy:.4f}**")

    # Make prediction based on user input
    user_prediction = (model.predict(input_scaled) > 0.5).astype("int32")

    # Display user prediction
    st.subheader("Predicted Diabetes Risk:")
    if user_prediction[0][0] == 1:
        st.error("⚠️ High Risk of Diabetes (1)")
    else:
        st.success("✅ Low Risk of Diabetes (0)")

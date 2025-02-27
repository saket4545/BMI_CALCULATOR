import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the saved pipeline
try:
    with open('bmi_pipeline.pkl', 'rb') as file:
        pipeline = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'bmi_pipeline.pkl' is in the directory.")
    st.stop()

# BMI Categories Mapping
categories = {
    0: "Extremely Weak",
    1: "Weak",
    2: "Normal",
    3: "Overweight",
    4: "Obesity",
    5: "Extreme Obesity"
}

# Streamlit UI
st.title("🏋️ BMI Prediction App")
st.write("Enter your details below to predict your BMI category.")

# User Inputs
gender = st.selectbox("Select Gender", ["Male", "Female"])
height = st.number_input("Enter your height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Enter your weight (kg)", min_value=30, max_value=250, value=70)

# BMI Calculation & Display
bmi = weight / ((height / 100) ** 2)
st.write(f"### Your BMI: {bmi:.2f}")

# Prediction Button
if st.button('Predict BMI Category'):
    try:
        # Prepare input data
        input_data = pd.DataFrame([[gender, height, weight]], columns=['Gender', 'Height', 'Weight'])
        prediction = pipeline.predict(input_data)
        category = categories.get(prediction[0], "Unknown")
        
        # Display Prediction
        st.success(f"### Your BMI category is: **{category}**")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")


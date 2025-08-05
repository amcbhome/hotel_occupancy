import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# --- Load the saved model and features ---
try:
    model = joblib.load('linear_regression_model.pkl')
    features = joblib.load('model_features.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'linear_regression_model.pkl' and 'model_features.pkl' are in the same directory.")
    st.stop()

# --- Streamlit UI ---
st.title("Hotel Occupancy Rate Predictor")
st.write("This app predicts the hotel occupancy rate based on various factors using a linear regression model.")

# --- User Input Section ---
st.header("Input Data for Prediction")

# Input for future date
future_date = st.date_input("Select a future date:")

# Manual inputs for other features
col1, col2 = st.columns(2)
with col1:
    nearby_event_attendees = st.number_input(
        "Nearby Event Attendees",
        min_value=0,
        value=0,
        step=1000
    )
with col2:
    average_room_rate = st.number_input(
        "Average Room Rate ($)",
        min_value=0.0,
        value=150.0,
        step=5.0
    )

# --- Prediction Logic ---
if st.button("Predict Occupancy Rate"):
    if future_date:
        # Process date to get 'Is_Weekend' and 'Is_Holiday_Period'
        day_of_week = future_date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        # Simple holiday period logic (e.g., fixed dates for holidays and summer)
        month = future_date.month
        day = future_date.day
        is_holiday_period = 0
        if (month == 12 and day >= 20) or (month == 1 and day <= 5) or (month in [7, 8]):
            is_holiday_period = 1

        # Create a DataFrame for the model
        input_data = pd.DataFrame([
            [is_weekend, is_holiday_period, nearby_event_attendees, average_room_rate]
        ], columns=features)

        # Make the prediction
        prediction = model.predict(input_data)
        predicted_occupancy = np.clip(prediction[0], 0, 1)

        # Display the prediction
        st.header("Prediction Result")
        st.success(f"The predicted occupancy rate for {future_date.strftime('%B %d, %Y')} is: **{predicted_occupancy:.2f}**")
    else:
        st.warning("Please select a valid date to get a prediction.")

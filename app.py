
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Cricket Pressure Predictor")

# Load trained model
model_path = Path(__file__).with_name("pressure_model.pkl")
model = joblib.load(model_path)

st.title("🏏 Cricket Pressure Predictor")
st.write("Predict batting pressure using Machine Learning")

st.subheader("Enter Match Details")

runs = st.number_input(
    "Runs Required", min_value=0, value=20
)

balls = st.number_input(
    "Balls Remaining", min_value=1, value=60
)

wickets = st.number_input(
    "Wickets Lost", min_value=0, max_value=10, value=2
)

if st.button("Predict Pressure"):

    required_rate = (runs * 6) / balls

    input_data = pd.DataFrame({
        "required_run_rate": [required_rate],
        "wickets_lost": [wickets]
    })

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    st.metric("Required Run Rate", round(required_rate, 2))

    if prediction == "Low":
        st.success("Pressure Level: LOW")

    elif prediction == "Medium":
        st.warning("Pressure Level: MEDIUM")

    else:
        st.error("Pressure Level: HIGH")

    st.write("Machine Learning Prediction:", prediction)

import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Wellbeing Predictor", page_icon="🩺")
st.title("🩺 Employee Wellbeing Prediction App")
st.markdown("### Predict your health status based on your lifestyle data.")

# ---- User Inputs ----
age = st.number_input("Age", min_value=18, max_value=70, value=25)
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=22.5)
sleep_hours = st.number_input("Average Sleep (hours)", min_value=0, max_value=12, value=7)
stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
exercise = st.selectbox("Exercise Frequency", ["None", "Occasional", "Regular"])

# ---- Encode Exercise ----
exercise_map = {"None": 0, "Occasional": 1, "Regular": 2}
exercise_encoded = exercise_map[exercise]

# ---- Feature Array ----
features = np.array([[age, bmi, sleep_hours, stress_level, exercise_encoded]])

# ---- Prediction ----
if st.button("🔍 Predict Health Status"):
    try:
        prediction = model.predict(features)
        st.success(f"Predicted Health Status: **{prediction[0]}**")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

st.markdown("---")
st.markdown("Made with ❤️ for Employee Wellbeing Research")

# app.py
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("salary_model.pkl")

st.set_page_config(page_title="💰 Data Science Salary Predictor", layout="centered")

st.title("💼 Data Science Salary Predictor")
st.write("Enter job details to estimate the annual salary in USD 💵")

# --- Sidebar ---
st.sidebar.header("Job Information")

# Friendly dropdowns
experience_options = {
    "Entry-level / Junior": "EN",
    "Mid-level / Intermediate": "MI",
    "Senior-level / Expert": "SE",
    "Executive-level / Director": "EX"
}

employment_options = {
    "Full-time": "FT",
    "Part-time": "PT",
    "Contract": "CT",
    "Freelance": "FL"
}

company_size_options = {
    "Small (less than 50 employees)": "S",
    "Medium (50–250 employees)": "M",
    "Large (250+ employees)": "L"
}

# User input
work_year = st.sidebar.selectbox("Work Year", [2020, 2021, 2022])
experience_label = st.sidebar.selectbox("Experience Level", list(experience_options.keys()))
employment_label = st.sidebar.selectbox("Employment Type", list(employment_options.keys()))
job_title = st.sidebar.text_input("Job Title", "Data Scientist")
employee_residence = st.sidebar.text_input("Employee Residence (country code)", "US")
remote_ratio = st.sidebar.slider("Remote Ratio (%)", 0, 100, 50)
company_location = st.sidebar.text_input("Company Location (country code)", "US")
company_size_label = st.sidebar.selectbox("Company Size", list(company_size_options.keys()))

# Map labels to model codes
experience_level = experience_options[experience_label]
employment_type = employment_options[employment_label]
company_size = company_size_options[company_size_label]

# --- Prepare data ---
input_data = pd.DataFrame({
    "work_year": [work_year],
    "experience_level": [experience_level],
    "employment_type": [employment_type],
    "job_title": [job_title],
    "employee_residence": [employee_residence],
    "remote_ratio": [remote_ratio],
    "company_location": [company_location],
    "company_size": [company_size]
})

# --- Predict ---
if st.button("💰 Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.success(f"💵 Estimated Salary: **${prediction:,.2f} USD**")

# --- Footer ---
st.markdown("---")
st.caption("Built using Streamlit and scikit-learn | Dataset: Kaggle Data Science Job Salaries")

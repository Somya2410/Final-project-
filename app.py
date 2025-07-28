
import streamlit as st
import pandas as pd
import pickle
from utils import load_data, preprocess, make_prediction

st.set_page_config(page_title="AttriVision", layout="wide")
st.title("🚨 Employee Attrition Prediction App")

uploaded_file = st.file_uploader("Upload HR Dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()

st.dataframe(df.head())

# Sidebar prediction inputs
st.sidebar.header("🔍 Predict for One Employee")
input_data = {
    "Age": st.sidebar.slider("Age", 18, 60, 30),
    "MonthlyIncome": st.sidebar.slider("Monthly Income", 1000, 20000, 5000),
    "JobSatisfaction": st.sidebar.slider("Job Satisfaction", 1, 4, 3),
    "OverTime": st.sidebar.selectbox("OverTime", ["Yes", "No"])
}

employee_df = pd.DataFrame([input_data])
pred, prob = make_prediction(employee_df)

st.sidebar.markdown(f"### Prediction: {'Yes' if pred == 1 else 'No'}")
st.sidebar.markdown(f"### Probability: {prob:.2f}")

# Visualizations (example placeholder)
st.header("📊 Attrition Dashboard")
st.bar_chart(df['JobRole'].value_counts())

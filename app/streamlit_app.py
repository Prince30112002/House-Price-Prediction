import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("🏠 House Price Prediction App")

# ----------------------
# Load trained models and columns
# ----------------------
lr_model = joblib.load("models/linear_regression_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
columns = joblib.load("models/columns.pkl")  # Training columns

st.sidebar.header("Enter House Features")

# ----------------------
# User Inputs
# ----------------------
lot_area = st.sidebar.number_input("Lot Area", min_value=500, max_value=20000, value=8450)
overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5)
year_built = st.sidebar.number_input("Year Built", min_value=1870, max_value=2025, value=2003)
total_bsmt_sf = st.sidebar.number_input("Total Basement SF", min_value=0, max_value=5000, value=856)
gr_liv_area = st.sidebar.number_input("Ground Living Area", min_value=300, max_value=4000, value=1710)
full_bath = st.sidebar.slider("Full Bath", 0, 4, 2)
garage_cars = st.sidebar.slider("Garage Cars", 0, 4, 2)

# ----------------------
# Prepare input for prediction
# ----------------------
input_dict = {
    "Lot Area": lot_area,
    "Overall Qual": overall_qual,
    "Year Built": year_built,
    "Total Bsmt SF": total_bsmt_sf,
    "Gr Liv Area": gr_liv_area,
    "Full Bath": full_bath,
    "Garage Cars": garage_cars
}

user_input = pd.DataFrame([input_dict])

# ----------------------
# Add missing columns & reorder to match training
# ----------------------
for col in columns:
    if col not in user_input.columns:
        user_input[col] = 0

user_input = user_input[columns]

# ----------------------
# Predict
# ----------------------
if st.button("Predict Price"):
    pred_lr = lr_model.predict(user_input)[0]
    pred_xgb = xgb_model.predict(user_input)[0]

    st.subheader("Predicted House Prices")
    st.write(f"Linear Regression: ${pred_lr:,.2f}")
    st.write(f"XGBoost: ${pred_xgb:,.2f}")

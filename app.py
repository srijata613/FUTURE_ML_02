import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# --- Load model and scaler ---
model = xgb.XGBClassifier()
model.load_model("xgb_model.json")  # your trained XGBoost model
scaler = joblib.load("scaler.pkl")  # your saved scaler

# --- Page setup ---
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict churn, or upload a CSV for batch predictions.")

# --- Option: Batch CSV Upload ---
uploaded_file = st.file_uploader("Upload CSV for batch predictions", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(df.head())
    
    # Make sure columns match your trained model features
    scaled_df = scaler.transform(df)
    predictions = model.predict(scaled_df)
    df["Churn_Prediction"] = predictions
    st.write("Predictions:")
    st.dataframe(df)
else:
    # --- Input fields ---
    st.subheader("Customer Details")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        age = st.slider("Age", 18, 80, 30)
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    with col2:
        balance = st.number_input("Balance", min_value=0.0, step=100.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_credit_card = st.selectbox("Has Credit Card", [0, 1])
        is_active = st.selectbox("Is Active Member", [0, 1])
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=1000.0)

    # --- Preprocess input ---
    input_features = np.zeros(19)  # match trained model feature count
    
    # Map numeric features
    input_features[3] = age
    input_features[4] = tenure
    input_features[5] = balance
    input_features[6] = num_products
    input_features[7] = has_credit_card
    input_features[8] = is_active
    input_features[9] = estimated_salary
    
    # One-hot encode gender
    input_features[0] = 1 if gender == "Male" else 0
    input_features[1] = 1 if gender == "Female" else 0
    
    # One-hot encode geography
    input_features[10] = 1 if geography == "France" else 0
    input_features[11] = 1 if geography == "Germany" else 0
    input_features[12] = 1 if geography == "Spain" else 0
    
    # Add any additional categorical feature mapping here if needed (13-18)
    
    # Scale input
    scaled_features = scaler.transform([input_features])
    
    # --- Prediction ---
    if st.button("Predict"):
        pred = model.predict(scaled_features)[0]
        prob = model.predict_proba(scaled_features)[0][1]  # probability of churn
        
        st.write(f"Churn Probability: **{prob*100:.2f}%**")
        if pred == 1:
            st.error("⚠️ The customer is likely to churn.")
        else:
            st.success("✅ The customer is likely to stay.")
        
        # --- Feature importance ---
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        feature_names = [
            "Gender_M", "Gender_F", "Other1", "Age", "Tenure", "Balance",
            "Num_Products", "Has_Credit", "Is_Active", "Salary",
            "France", "Germany", "Spain", "Other2", "Other3", "Other4", "Other5", "Other6", "Other7"
        ]
        plt.figure(figsize=(8,6))
        plt.barh(feature_names, importances)
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        st.pyplot(plt)

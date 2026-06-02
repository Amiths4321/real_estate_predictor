import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load('real_estate_model.pkl')

st.title("🏠 Real Estate Price Predictor")
st.write("Enter property details to get an estimated price")

# ---- Input Fields ----
area = st.number_input("Area (sq ft)", min_value=300, max_value=10000, value=1200)
bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4])
age = st.slider("Property Age (years)", 0, 50, 5)
location_score = st.slider("Location Score (1-10)", 1, 10, 7)

# ---- Predict Button ----
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'area_sqft': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'age_years': [age],
        'location_score': [location_score]
    })
    price = model.predict(input_data)[0]
    st.success(f"💰 Estimated Price: ₹ {price:,.2f} Lakhs")
    st.balloons()
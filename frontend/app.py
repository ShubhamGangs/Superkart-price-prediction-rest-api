import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Set the title of the Streamlit app
st.title("SuperKart Store Sales Forecasting")

# Section for online prediction
st.subheader("Online Prediction")

# Collect user input for property features
product_weight = st.number_input("Product Weight (kg)", min_value=0.0, value=12.65)
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.068, format="%.4f")
product_mrp = st.number_input("Product MRP ($)", min_value=0.0, value=147.0)
store_age_years = st.number_input("Store Age (Years)", min_value=1, step=1, value=17)

product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_type = st.selectbox("Product Type", ["Fruits and Vegetables", "Snack Foods", "Frozen Foods", "Dairy", "Household", "Baking Goods", "Canned", "Health and Hygiene", "Meat", "Soft Drinks", "Breads", "Hard Drinks", "Others", "Starchy Foods", "Breakfast", "Seafood"])
store_size = st.selectbox("Store Size", ["Medium", "High", "Small"])
store_location_city_type = st.selectbox("Store Location City Type", ["Tier 2", "Tier 1", "Tier 3"])
store_type = st.selectbox("Store Type", ["Supermarket Type2", "Supermarket Type1", "Departmental Store", "Food Mart"])
store_age_category = st.selectbox("Store Age Category", ["New", "Medium", "Old"])

# Convert user input into a DataFrame
input_data = pd.DataFrame([{
    'Product_Weight': product_weight,
    'Product_Allocated_Area': product_allocated_area,
    'Product_MRP': product_mrp,
    'Store_Age_Years': store_age_years,
    'Product_Sugar_Content': product_sugar_content,
    'Product_Type': product_type,
    'Store_Size': store_size,
    'Store_Location_City_Type': store_location_city_type,
    'Store_Type': store_type,
    'Store_Age_Category': store_age_category
}])

# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/predict", json=input_data.to_dict(orient='records')[0])  # Send data to Flask API
    if response.status_code == 200:
        prediction = response.json()['Predicted Sales (in dollars)']
        st.success(f"Predicted Product Store Sales (in dollars): {prediction}")
    else:
        st.error("Unable to connect to the prediction API.")

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files={"file": uploaded_file})  # Send file to Flask API
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)  # Display the predictions
        else:
            st.error("Unable to connect to the prediction API.")

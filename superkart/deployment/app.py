import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="rkpworks/superkart-sales-model",
    filename="best_superkart_sales_model.joblib"
)
model = joblib.load(model_path)

# ---- Streamlit UI ----
st.title("SuperKart — Product Store Sales Prediction")
st.write("Enter product and store details below to predict the total sales for a product at a given store.")

# Collect user inputs
Product_Weight = st.number_input("Product Weight (kg)", min_value=0.0, max_value=50.0, value=12.0)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular"])
Product_Allocated_Area = st.number_input("Product Allocated Area (shelf space proportion)", min_value=0.0, max_value=1.0, value=0.05, format="%.4f")
Product_Type = st.selectbox("Product Type", [
    "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables",
    "Household", "Baking Goods", "Snack Foods", "Frozen Foods",
    "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned",
    "Breads", "Starchy Foods", "Others", "Seafood"
])
Product_MRP = st.number_input("Product MRP (Maximum Retail Price)", min_value=0.0, max_value=300.0, value=140.0)
Store_Establishment_Year = st.number_input("Store Establishment Year", min_value=1985, max_value=2009, value=2000)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", [
    "Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"
])

# Build input dataframe
input_data = pd.DataFrame([{
    'Product_Weight': Product_Weight,
    'Product_Allocated_Area': Product_Allocated_Area,
    'Product_MRP': Product_MRP,
    'Store_Establishment_Year': Store_Establishment_Year,
    'Product_Sugar_Content': Product_Sugar_Content,
    'Product_Type': Product_Type,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type
}])

# Predict
if st.button("Predict Sales"):
    predicted_sales = model.predict(input_data)[0]
    st.success(f"Predicted Product Store Sales: **${predicted_sales:,.2f}**")

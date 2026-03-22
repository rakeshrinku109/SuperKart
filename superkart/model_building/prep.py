import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# Authenticate with Hugging Face
api = HfApi(token=os.getenv("HF_TOKEN"))

# --------------------------------
# Load the dataset directly from the Hugging Face data space
# --------------------------------
DATASET_PATH = "hf://datasets/rkpworks/SuperKart-Sales/SuperKart.csv"
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded. Shape: {df.shape}")

# --------------------------------
# Data Cleaning
# --------------------------------

# Remove identifier columns that have no predictive value
df.drop(columns=['Product_Id', 'Store_Id'], inplace=True)

# Fill missing Product_Weight values with the median
df['Product_Weight'].fillna(df['Product_Weight'].median(), inplace=True)

# Fill missing Store_Size values with the mode
df['Store_Size'].fillna(df['Store_Size'].mode()[0], inplace=True)

# Standardise inconsistent sugar content labels
df['Product_Sugar_Content'] = df['Product_Sugar_Content'].replace({
    'low fat': 'Low Sugar',
    'LF': 'Low Sugar',
    'reg': 'Regular'
})

print(f"Cleaned dataset shape: {df.shape}")

# --------------------------------
# Define target and features
# --------------------------------
target = 'Product_Store_Sales_Total'

numeric_features = [
    'Product_Weight',            # Weight of the product
    'Product_Allocated_Area',    # Display area allocated in the store
    'Product_MRP',               # Maximum Retail Price
    'Store_Establishment_Year'   # Year the store was established
]

categorical_features = [
    'Product_Sugar_Content',      # Low Sugar or Regular
    'Product_Type',               # Product category (16 types)
    'Store_Size',                 # Small, Medium, or High
    'Store_Location_City_Type',   # Tier 1, 2, or 3
    'Store_Type'                  # Grocery Store, Supermarket Type1/2/3
]

X = df[numeric_features + categorical_features]
y = df[target]

# --------------------------------
# Split into training and test sets
# --------------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size: {Xtrain.shape[0]}, Test size: {Xtest.shape[0]}")

# Save splits locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload splits to Hugging Face
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="rkpworks/SuperKart-Sales",
        repo_type="dataset",
    )

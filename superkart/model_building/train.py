import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("superkart-sales-production")

api = HfApi()

# ---- Load train and test data from Hugging Face ----
Xtrain = pd.read_csv("hf://datasets/rkpworks/SuperKart-Sales/Xtrain.csv")
Xtest  = pd.read_csv("hf://datasets/rkpworks/SuperKart-Sales/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/rkpworks/SuperKart-Sales/ytrain.csv")
ytest  = pd.read_csv("hf://datasets/rkpworks/SuperKart-Sales/ytest.csv")

print(f"Data loaded — Xtrain: {Xtrain.shape}, Xtest: {Xtest.shape}")

# ---- Feature definitions ----
numeric_features = [
    'Product_Weight',            # Weight of the product
    'Product_Allocated_Area',    # Display area allocated in the store
    'Product_MRP',               # Maximum Retail Price
    'Store_Establishment_Year'   # Year the store was established
]

categorical_features = [
    'Product_Sugar_Content',      # Low Sugar or Regular
    'Product_Type',               # Product category
    'Store_Size',                 # Small, Medium, or High
    'Store_Location_City_Type',   # City tier
    'Store_Type'                  # Store format
]

# ---- Preprocessing ----
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# XGBoost Regressor for continuous sales prediction
xgb_model = xgb.XGBRegressor(random_state=42)

# Extended hyperparameter grid for production
param_grid = {
    'xgbregressor__n_estimators': [50, 100, 150, 200],
    'xgbregressor__max_depth': [3, 5, 7],
    'xgbregressor__colsample_bytree': [0.5, 0.7, 0.9],
    'xgbregressor__colsample_bylevel': [0.5, 0.7],
    'xgbregressor__learning_rate': [0.01, 0.05, 0.1],
    'xgbregressor__reg_lambda': [0.3, 0.5, 0.7],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

# ---- MLflow-tracked training ----
with mlflow.start_run():
    grid_search = GridSearchCV(
        model_pipeline, param_grid, cv=5,
        scoring='r2', n_jobs=-1
    )
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_cv_r2", mean_score)
            mlflow.log_metric("std_cv_r2", std_score)

    # Log the best parameters on the parent run
    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # Evaluate on training data
    y_pred_train = best_model.predict(Xtrain)
    train_rmse = np.sqrt(mean_squared_error(ytrain, y_pred_train))
    train_mae = mean_absolute_error(ytrain, y_pred_train)
    train_r2 = r2_score(ytrain, y_pred_train)

    # Evaluate on test data
    y_pred_test = best_model.predict(Xtest)
    test_rmse = np.sqrt(mean_squared_error(ytest, y_pred_test))
    test_mae = mean_absolute_error(ytest, y_pred_test)
    test_r2 = r2_score(ytest, y_pred_test)

    # Log regression metrics
    mlflow.log_metrics({
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2
    })

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Train — RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R2: {train_r2:.4f}")
    print(f"Test  — RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R2: {test_r2:.4f}")

    # Save model locally
    model_path = "best_superkart_sales_model.joblib"
    joblib.dump(best_model, model_path)

    # Log as MLflow artefact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact: {model_path}")

    # ---- Register model on Hugging Face Model Hub ----
    MODEL_REPO = "rkpworks/superkart-sales-model"

    try:
        api.repo_info(repo_id=MODEL_REPO, repo_type="model")
        print(f"Model repo '{MODEL_REPO}' already exists.")
    except RepositoryNotFoundError:
        print(f"Creating model repo '{MODEL_REPO}'...")
        create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)
        print(f"Model repo '{MODEL_REPO}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_superkart_sales_model.joblib",
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    print(f"Model uploaded to Hugging Face: {MODEL_REPO}")

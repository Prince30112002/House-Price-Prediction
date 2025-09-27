import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# XGBoost import
try:
    from xgboost import XGBRegressor
except ImportError:
    print("❌ XGBoost not installed! Install with: pip install xgboost")
    exit()

print("🚀 Model Training Script started...")

# Paths
processed_path = "data/processed/ames_cleaned.csv"
models_dir = "models/"

# Check if processed dataset exists
if not os.path.exists(processed_path):
    print(f"❌ ERROR: Processed dataset not found at {processed_path}")
    exit()

# Create models directory if not exists
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Load dataset
df = pd.read_csv(processed_path)
print("✅ Cleaned dataset loaded. Shape:", df.shape)

# Features & target
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Linear Regression ----------------
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# RMSE compatible with older sklearn versions
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("\n📌 Linear Regression Metrics:")
print(f"RMSE: {rmse_lr:.2f}, R²: {r2_lr:.4f}")

# Save Linear Regression model
lr_path = os.path.join(models_dir, "linear_regression_model.pkl")
joblib.dump(lr, lr_path)
print(f"✅ Linear Regression model saved at {lr_path}")

# ---------------- XGBoost Regressor ----------------
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbosity=0
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# RMSE compatible with older sklearn versions
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print("\n📌 XGBoost Regressor Metrics:")
print(f"RMSE: {rmse_xgb:.2f}, R²: {r2_xgb:.4f}")

# Save XGBoost model
xgb_path = os.path.join(models_dir, "xgb_model.pkl")
joblib.dump(xgb, xgb_path)
print(f"✅ XGBoost model saved at {xgb_path}")

print("\n🎯 Model Training Script finished successfully!")

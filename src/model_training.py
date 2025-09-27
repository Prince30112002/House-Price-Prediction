import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import joblib
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# Create models folder if not exists
if not os.path.exists("models"):
    os.makedirs("models")

# Load processed data
data = pd.read_csv("data/processed/ames_cleaned.csv")
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------
# Linear Regression
# ----------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print("Linear Regression RMSE:", rmse_lr)

# ----------------------
# XGBoost Regressor
# ----------------------
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print("XGBoost RMSE:", rmse_xgb)

# ----------------------
# Save models using joblib
# ----------------------
joblib.dump(lr, "models/linear_regression_model.pkl")
joblib.dump(xgb_model, "models/xgb_model.pkl")
print("✅ Models saved successfully in 'models/' folder")

# ----------------------
# Save training columns
# ----------------------
joblib.dump(X_train.columns, "models/columns.pkl")
print("✅ Training feature columns saved in 'models/columns.pkl'")

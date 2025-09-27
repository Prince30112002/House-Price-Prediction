import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

print("🚀 Feature Engineering Script started...")

# Paths
raw_path = "data/raw/AmesHousing.csv"
processed_dir = "data/processed/"
processed_path = os.path.join(processed_dir, "ames_cleaned.csv")

# Check if raw dataset exists
if not os.path.exists(raw_path):
    print(f"❌ ERROR: Dataset not found at {raw_path}")
    exit()

# Create processed directory if not exists
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

# Load dataset
df = pd.read_csv(raw_path)
print("✅ Raw dataset loaded. Shape:", df.shape)

# Drop ID/order columns if present
drop_cols = [col for col in ["Order", "PID"] if col in df.columns]
df = df.drop(columns=drop_cols)

# Separate features
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

print(f"Numerical columns: {len(num_cols)}, Categorical columns: {len(cat_cols)}")

# Handle missing values
num_imputer = SimpleImputer(strategy="median")
X[num_cols] = num_imputer.fit_transform(X[num_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

# Encode categorical variables
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_encoded = pd.DataFrame(
    encoder.fit_transform(X[cat_cols]),
    columns=encoder.get_feature_names_out(cat_cols)
)

# Combine numerical + encoded categorical
X_final = pd.concat([X[num_cols].reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)
X_final["SalePrice"] = y.reset_index(drop=True)

# Save cleaned dataset
X_final.to_csv(processed_path, index=False)
print(f"✅ Cleaned dataset saved at {processed_path}")
print("🎯 Feature Engineering Script finished successfully!")

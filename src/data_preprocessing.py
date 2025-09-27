import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data/raw/AmesHousing.csv")

print("Dataset Shape:", data.shape)
print("Columns:", data.columns[:10])  # first 10 columns preview

# Example split
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

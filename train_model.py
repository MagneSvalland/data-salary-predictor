# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# === Load dataset ===
DATA_PATH = os.path.join("data", "ds_salaries.csv")
df = pd.read_csv(DATA_PATH)

# === Select features and target ===
X = df[[
    "work_year", "experience_level", "employment_type",
    "job_title", "employee_residence", "remote_ratio",
    "company_location", "company_size"
]]
y = df["salary_in_usd"]

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Preprocess categorical features ===
categorical_cols = ["experience_level", "employment_type", "job_title",
                    "employee_residence", "company_location", "company_size"]
numeric_cols = ["work_year", "remote_ratio"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# === Build model pipeline ===
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# === Train model ===
model.fit(X_train, y_train)

# === Evaluate model ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# === Save trained model ===
joblib.dump(model, "salary_model.pkl")
print("💾 Model saved to salary_model.pkl")

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score

# === Load Dataset ===
df = pd.read_csv("INPUT YOUR FILE PATH HERE")

# === Drop non-numeric or irrelevant columns ===
df = df.drop(columns=["town/city", "railway_period", "real_price"], errors="ignore")

# === Define Features and Target ===
one_hot_cols = [col for col in df.columns if col.startswith("railway_")]
feature_cols = [
    'log_distance_to_station',
    'business_count',
    'log_interaction',
    'year_of_transaction',
    *one_hot_cols
]
target_col = 'log_real_price'
X = df[feature_cols]
y = df[target_col]

# === Define Models ===
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

# === Cross-Validation ===
cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    rmse_scores = -cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    mae_scores = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

    results.append({
        "Model": name,
        "RMSE Mean": round(rmse_scores.mean(), 4),
        "RMSE Std": round(rmse_scores.std(), 4),
        "MAE Mean": round(mae_scores.mean(), 4),
        "R² Mean": round(r2_scores.mean(), 4)
    })

# === Save Results ===
output_path = "INPUT YOUR FILE PATH HERE"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
pd.DataFrame(results).to_csv(output_path, index=False)

# === Display Results ===
print("Cross-validation completed and saved to:")
print(output_path)
print(pd.DataFrame(results))

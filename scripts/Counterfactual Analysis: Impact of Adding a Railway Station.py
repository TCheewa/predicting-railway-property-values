import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import os

# === Load Dataset ===
df = pd.read_csv("INPUT YOUR FILE PATH HERE")

# === Features and Target ===
feature_cols = [
    'log_distance_to_station', 'business_count', 'log_interaction',
    'year_of_transaction', 'railway_Control-NoStation', 'railway_Control-Station',
    'railway_Post', 'railway_Pre'
]
target_col = 'log_real_price'

# === Train LightGBM model on entire dataset ===
X = df[feature_cols]
y = df[target_col]

model_lgbm = LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
model_lgbm.fit(X, y)

# === Function to simulate counterfactual ===
def simulate_counterfactual_lgbm(town_name, save_path):
    subset = df[df['town/city'].str.upper() == town_name.upper()].copy()
    if subset.empty:
        print(f"⚠️ Town '{town_name}' not found.")
        return

    # Set counterfactual status
    subset['railway_Control-NoStation'] = 0
    subset['railway_Post'] = 1

    # Predict
    X_counter = subset[feature_cols]
    log_pred = model_lgbm.predict(X_counter)
    subset['predicted_price_if_has_station'] = np.expm1(log_pred)
    subset['real_price'] = np.expm1(subset['log_real_price'])
    subset['price_diff'] = subset['predicted_price_if_has_station'] - subset['real_price']

    # Save
    subset[['postcode', 'date_of_transfer', 'real_price', 'predicted_price_if_has_station', 'price_diff']].to_csv(save_path, index=False)
    print(f"✅ Counterfactual saved: {save_path}")

# === Run for both towns ===
simulate_counterfactual_lgbm("WISBECH", "INPUT YOUR FILE PATH HERE")
simulate_counterfactual_lgbm("RUSHDEN", "INPUT YOUR FILE PATH HERE")

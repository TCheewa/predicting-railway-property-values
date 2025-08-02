import pandas as pd
import numpy as np
import os

# === Paths ===
input_path = "INPUT YOUR FILE PATH HERE"
output_path = "INPUT YOUR FILE PATH HERE"

# === Load Data ===
df = pd.read_csv(input_path, low_memory=False)

# === Basic Cleaning ===
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['real_price'] = pd.to_numeric(df['real_price'], errors='coerce')
df['distance_to_station'] = pd.to_numeric(df['distance_to_station'], errors='coerce')
df['business_count'] = pd.to_numeric(df['business_count'].astype(str).str.replace(',', ''), errors='coerce')
df['year_of_transaction'] = pd.to_datetime(df['date_of_transfer'], errors='coerce').dt.year
df['railway_period'] = df['railway_period'].fillna('Unknown').astype(str)
df['town/city'] = df['town/city'].astype(str).str.upper()

# === Drop rows with key missing values ===
df = df.dropna(subset=[
    'real_price', 'distance_to_station', 'business_count', 'year_of_transaction'
])
df = df[df['real_price'] < 2_000_000]

# === Feature Engineering ===
df['log_real_price'] = np.log1p(df['real_price'])
df['log_distance_to_station'] = np.log1p(df['distance_to_station'])
df['interaction'] = df['distance_to_station'] * df['business_count']
df['log_interaction'] = np.log1p(df['interaction'])

# Optional: Create time-based bins (e.g. before/after 2015)
df['period_group'] = pd.cut(df['year_of_transaction'],
                            bins=[1994, 2005, 2015, 2025],
                            labels=['<2005', '2005–2015', '2016–2024'])

# === Encode railway_period (One-hot) ===
railway_dummies = pd.get_dummies(df['railway_period'], prefix='railway')
df = pd.concat([df, railway_dummies], axis=1)

# === Final Columns Selection (for model) ===
feature_cols = [
    'log_distance_to_station',
    'business_count',
    'log_interaction',
    'year_of_transaction',
    # 'period_group',  # optionally use categorical encoding if needed
    *railway_dummies.columns
]
target_col = 'log_real_price'

# === Save Final Dataset ===
df_model = df[feature_cols + [
    target_col,
    'real_price',
    'town/city',
    'railway_period',
    'postcode',
    'date_of_transfer'
]]
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_model.to_csv(output_path, index=False)

print(f"Final model-ready dataset saved to:\n{output_path}")

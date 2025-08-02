import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import os

# === Paths ===
input_path = "INPUT YOUR FILE PATH HERE"
output_path = "INPUT YOUR FILE PATH HERE"

# === Load data ===
df = pd.read_csv(input_path, low_memory=False)
df['town/city'] = df['town/city'].str.upper()

# === Manual caps: extreme filter ===
df = df[
    df['real_price'].notna() &
    df['distance_to_station'].notna() &
    df['business_count'].notna() &
    (df['real_price'] < 2_000_000) &
    (df['distance_to_station'] < 20000)
].copy()

# === IQR Filter function ===
def iqr_filter(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return series.between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# === Apply IQR filtering ===
iqr_mask = (
    iqr_filter(df['real_price']) &
    iqr_filter(df['distance_to_station']) &
    iqr_filter(df['business_count'])
)
df_iqr = df[iqr_mask].copy()

# === Winsorization (soft cap) ===
df_iqr['real_price_win'] = winsorize(df_iqr['real_price'], limits=[0.01, 0.01])
df_iqr['distance_win'] = winsorize(df_iqr['distance_to_station'], limits=[0.01, 0.01])
df_iqr['business_count_win'] = winsorize(df_iqr['business_count'], limits=[0.01, 0.01])

# === Log-transform ===
df_iqr['log_price'] = np.log1p(df_iqr['real_price_win'])
df_iqr['log_distance'] = np.log1p(df_iqr['distance_win'])
df_iqr['log_business'] = np.log1p(df_iqr['business_count_win'])

# === Save cleaned and transformed data ===
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_iqr.to_csv(output_path, index=False)
print(f"Cleaned + winsorized + log-transformed dataset saved to:\n{output_path}")

import pandas as pd
import numpy as np

# Load dataset
input_file = "INPUT YOUR FILE PATH HERE"
df = pd.read_csv(input_file, low_memory=False)

df['real_price'] = pd.to_numeric(df['real_price'], errors='coerce')
df['business_count'] = pd.to_numeric(df['business_count'].astype(str).str.replace(',', ''), errors='coerce')
df['distance_to_station'] = pd.to_numeric(df['distance_to_station'], errors='coerce')

df['business_count'] = df['business_count'].fillna(0)

df_cleaned = df.dropna(subset=['real_price', 'distance_to_station'])

# Create new interaction
df_cleaned['distance_times_business'] = df_cleaned['distance_to_station'] * df_cleaned['business_count']
df_cleaned['log_interaction'] = np.log1p(df_cleaned['distance_times_business'])

# Create dummy variables
df_cleaned['near_station'] = (df_cleaned['distance_to_station'] < 2000).astype(int)
df_cleaned['business_dense'] = (df_cleaned['business_count'] > df_cleaned['business_count'].median()).astype(int)
df_cleaned['target_area'] = (df_cleaned['near_station'] & df_cleaned['business_dense']).astype(int)

# Save File
output_file = "INPUT YOUR FILE PATH HERE"
df_cleaned.to_csv(output_file, index=False)
print(f"Dataset with imputed business_count saved to:\n{output_file}")

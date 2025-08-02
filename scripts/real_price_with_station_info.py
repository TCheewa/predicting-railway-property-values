import pandas as pd

# === File paths ===
property_file = "INPUT YOUR FILE PATH HERE"
nearest_station_file = "INPUT YOUR FILE PATH HERE"
output_file = "INPUT YOUR FILE PATH HERE"

# === Load datasets ===
df_property = pd.read_csv(property_file, low_memory=False)
df_station = pd.read_csv(nearest_station_file)

# === Standardize postcode format: remove spaces + uppercase
df_property['postcode_clean'] = df_property['postcode'].astype(str).str.replace(" ", "").str.upper()
df_station['postcode_clean'] = df_station['postcode'].astype(str).str.replace(" ", "").str.upper()

# === Merge on cleaned postcode
df_merged = df_property.merge(
    df_station[['postcode_clean', 'nearest_station_name', 'station_lat', 'station_lon',
                'station_creation_year', 'distance_to_station_km']],
    on='postcode_clean',
    how='left'
)

# === Manage missing station info
df_merged['nearest_station_name'] = df_merged['nearest_station_name'].fillna('NO_STATION')
df_merged['distance_to_station_km'] = df_merged['distance_to_station_km'].fillna(-1)
df_merged['station_creation_year'] = df_merged['station_creation_year'].fillna(-1)

# === Calculate distance to KM
df_merged['distance_to_station'] = df_merged['distance_to_station_km'] * 1000

# === Drop columns that not use 
df_merged.drop(columns=['postcode_clean', 'distance_to_station_km'], inplace=True)

# Check record that don't have nearest station
print("\n record that don't have nearest station:")
town_col = 'town/city' if 'town/city' in df_merged.columns else 'town'
if town_col in df_merged.columns:
    print(df_merged[df_merged['nearest_station_name'] == 'NO_STATION'][town_col].value_counts())
else:
    print("Do not found column 'town' or 'town/city' in dataset")

# === Save final output
df_merged.to_csv(output_file, index=False)
print(f"\n Merged dataset saved to:\n{output_file}")

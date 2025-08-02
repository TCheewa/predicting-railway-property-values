import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

# === File Paths ===
postcode_file = "INPUT YOUR FILE PATH HERE"
station_file = "INPUT YOUR FILE PATH HERE"
output_file = "INPUT YOUR FILE PATH HERE"

# === Load Data ===
df_post = pd.read_csv(postcode_file)
df_station = pd.read_csv(station_file)

# === Rename Columns for Uniformity ===
df_post = df_post.rename(columns={'Postcode': 'postcode', 'Latitude': 'prop_lat', 'Longitude': 'prop_lon'})
df_station = df_station.rename(columns={'CommonName': 'station_name', 'Latitude': 'station_lat', 'Longitude': 'station_lon'})

# === Drop missing coordinates ===
df_post = df_post.dropna(subset=['prop_lat', 'prop_lon'])
df_station = df_station.dropna(subset=['station_lat', 'station_lon', 'creation_year'])

# === Create coordinate tuples and convert to radians ===
df_post['coords'] = list(zip(df_post['prop_lat'], df_post['prop_lon']))
df_station['coords'] = list(zip(df_station['station_lat'], df_station['station_lon']))
post_radians = np.radians(df_post['coords'].tolist())
station_radians = np.radians(df_station['coords'].tolist())

# === Find nearest station using BallTree ===
tree = BallTree(station_radians, metric='haversine')
distances, indices = tree.query(post_radians, k=1)
df_post['distance_to_station_km'] = distances.flatten() * 6371  # Convert radians to km

# === Add station info to postcodes ===
nearest_stations = df_station.iloc[indices.flatten()].reset_index(drop=True)
df_post['nearest_station_name'] = nearest_stations['station_name']
df_post['station_lat'] = nearest_stations['station_lat']
df_post['station_lon'] = nearest_stations['station_lon']
df_post['station_creation_year'] = nearest_stations['creation_year']

# === Save to CSV ===
df_post.to_csv(output_file, index=False)
print(f"Output saved to: {output_file}")

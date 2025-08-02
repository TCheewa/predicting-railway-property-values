import pandas as pd

naptan_file = "INPUT YOUR FILE PATH HERE"
naptan = pd.read_csv(naptan_file, low_memory=False)

# Filter only railway stations with StopType 'RSE' or 'RLY'
rail_stations = naptan[naptan['StopType'].isin(['RSE', 'RLY'])].copy()

# Extract creation year from CreationDateTime column
rail_stations['creation_year'] = pd.to_datetime(
    rail_stations['CreationDateTime'], errors='coerce').dt.year

# Filter status: only active stations
rail_stations = rail_stations[
    rail_stations['Status'].str.lower().isin(['active', 'act'])
]

# Drop rows missing latitude, longitude, or creation_year
rail_stations = rail_stations.dropna(subset=['Latitude', 'Longitude', 'creation_year'])

# Keep useful columns only
rail_stations = rail_stations[['CommonName', 'LocalityName',
                               'Town', 'Latitude', 'Longitude', 'creation_year']]

# Drop duplicates based on station name
rail_stations = rail_stations.drop_duplicates(subset=['CommonName'])

# Save output file
output_path = "INPUT YOUR FILE PATH HERE"
rail_stations.to_csv(output_path, index=False)

print(f"rail_stations_with_year.csv saved to {output_path}")

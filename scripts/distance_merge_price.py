import pandas as pd
from geopy.distance import geodesic

# === Paths ===
property_file = "INPUT YOUR FILE PATH HERE"
station_file = "INPUT YOUR FILE PATH HERE"
output_file = "INPUT YOUR FILE PATH HERE"

# === Load datasets ===
properties = pd.read_csv(property_file)
stations = pd.read_csv(station_file)

# === Convert coordinates to numeric and drop invalid rows ===
for df in [properties, stations]:
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# === Extract year of transaction ===
properties['year_of_transaction'] = pd.to_datetime(properties['date_of_transfer'], errors='coerce').dt.year

# === Drop properties without a valid year (if any) ===
properties = properties.dropna(subset=['year_of_transaction'])

# === Ensure station creation_year is numeric ===
stations['creation_year'] = pd.to_numeric(stations['creation_year'], errors='coerce')
stations = stations.dropna(subset=['creation_year'])

# === Define distance calculation function ===
def get_nearest_station_distance(row):
    year = row['year_of_transaction']
    prop_coord = (row['Latitude'], row['Longitude'])

    # Filter stations that existed at that year
    valid_stations = stations[stations['creation_year'] <= year]

    if valid_stations.empty:
        return None  # No station existed yet

    try:
        distances = valid_stations.apply(
            lambda station: geodesic(prop_coord, (station['Latitude'], station['Longitude'])).meters,
            axis=1
        )
        return distances.min()
    except Exception as e:
        print(f"Error on row {row.name}: {e}")
        return None

# === Apply distance calculation ===
print("Calculating time-aware distance to nearest station...")
properties['distance_to_station'] = properties.apply(get_nearest_station_distance, axis=1)

# === Save the result ===
properties.to_csv(output_file, index=False)
print(f"Done: Distance column added and saved to:\n{output_file}")

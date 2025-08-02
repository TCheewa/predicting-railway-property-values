import pandas as pd
import os
from pyproj import Transformer

# Paths
codepoint_folder = "INPUT YOUR FILE PATH HERE"
property_file = "INPUT YOUR FILE PATH HERE"
codepoint_output = "INPUT YOUR FILE PATH HERE"
final_output = "INPUT YOUR FILE PATH HERE"

# Read and combine Code-Point Open CSV files
df_list = []

for file in os.listdir(codepoint_folder):
    if file.endswith(".csv"):
        path = os.path.join(codepoint_folder, file)
        df = pd.read_csv(path, header=None, encoding='latin1')  # safer encoding
        df = df[[0, 2, 3]]  # Columns: Postcode, Easting, Northing
        df.columns = ['Postcode', 'Easting', 'Northing']
        df_list.append(df)

codepoint = pd.concat(df_list, ignore_index=True)

# Convert Easting/Northing to Latitude/Longitude
transformer = Transformer.from_crs("epsg:27700", "epsg:4326", always_xy=True)
codepoint[['Longitude', 'Latitude']] = codepoint.apply(
    lambda row: pd.Series(transformer.transform(row['Easting'], row['Northing'])),
    axis=1
)

# Save full Code-Point data with lat/lon
codepoint.to_csv(codepoint_output, index=False)
print(f"Saved full codepoint with lat/lon to: {codepoint_output}")

# Load enriched property dataset
properties = pd.read_csv(property_file)

# Clean and standardise postcodes
properties['postcode'] = properties['postcode'].astype(str).str.strip().str.replace(" ", "").str.upper()
codepoint['Postcode'] = codepoint['Postcode'].astype(str).str.strip().str.replace(" ", "").str.upper()

# Drop rows with missing postcodes
properties = properties[properties['postcode'].notna()]

# Merge lat/lon into property dataset
merged = pd.merge(properties, codepoint, left_on='postcode', right_on='Postcode', how='left')
merged = merged.drop(columns=['Postcode'])

# Save the merged dataset
merged.to_csv(final_output, index=False)
print(f"Final property dataset with lat/lon saved to: {final_output}")

# Extra: Summary
missing_coords = merged[['Latitude', 'Longitude']].isna().sum().sum()
print(f"Properties missing lat/lon: {missing_coords} rows")

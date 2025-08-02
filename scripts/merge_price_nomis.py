import pandas as pd
import os

# === File paths ===
property_file = "INPUT YOUR FILE PATH HERE"
business_file = "INPUT YOUR FILE PATH HERE"
output_file = "INPUT YOUR FILE PATH HERE"

# === Load datasets ===
properties = pd.read_csv(property_file)
business = pd.read_csv(business_file)

# === Convert date and extract year ===
properties['date_of_transfer'] = pd.to_datetime(properties['date_of_transfer'], errors='coerce')
properties['year_of_transaction'] = properties['date_of_transfer'].dt.year

# === Melt business dataset: wide → long format ===
business = business.melt(id_vars=business.columns[0], var_name='year', value_name='business_count')
business = business.rename(columns={business.columns[0]: 'district'})
business['year'] = business['year'].astype(int)

# === Clean and standardize district text ===
properties['district'] = properties['district'].astype(str).str.strip().str.upper()
business['district'] = business['district'].astype(str).str.strip().str.upper()

# === Boundary & correction mapping (to match Nomis districts) ===
properties['district'] = properties['district'].replace({
    'CORBY': 'NORTH NORTHAMPTONSHIRE',
    'KETTERING': 'NORTH NORTHAMPTONSHIRE',
    'EAST NORTHAMPTONSHIRE': 'NORTH NORTHAMPTONSHIRE',
    'RUSHDEN': 'NORTH NORTHAMPTONSHIRE',  # 🆕

    'NORTHAMPTON': 'WEST NORTHAMPTONSHIRE',
    'SOUTH NORTHAMPTONSHIRE': 'WEST NORTHAMPTONSHIRE',

    'AYLESBURY VALE': 'BUCKINGHAMSHIRE',
    'BUCKINGHAMSHIRE': 'CHERWELL',  # for property data to align

    'WISBECH': 'FENLAND',
    "KING'S LYNN AND WEST NORFOLK": 'FENLAND',

    'REDDITCH': 'REDDITCH',

    'WARWICK': 'WARWICK',
    'KENILWORTH': 'WARWICK',

    'DAVENTRY': 'NORTH NORTHAMPTONSHIRE',
    'WELLINGBOROUGH': 'NORTH NORTHAMPTONSHIRE'
})

# === Merge datasets on ['district', 'year'] ===
final = properties.merge(
    business,
    left_on=['district', 'year_of_transaction'],
    right_on=['district', 'year'],
    how='left'
)

# === Fill missing business_count with 0 ===
final['business_count'] = final['business_count'].fillna(0)

# === Drop duplicate column
final = final.drop(columns=['year'])

# === Save final merged dataset
os.makedirs(os.path.dirname(output_file), exist_ok=True)
final.to_csv(output_file, index=False)

print("complete: final_enriched_dataset.csv saved")

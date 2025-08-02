import pandas as pd
import os

# === Paths ===
input_path = "INPUT YOUR FILE PATH HERE"
output_path = "INPUT YOUR FILE PATH HERE"
# === Column Definitions & Load ===
columns = [
    'transaction_unique_identifier','price','date_of_transfer','postcode',
    'property_type','old_new','duration','paon','saon','street','locality',
    'town/city','district','county','ppd_category_type','record_status'
]
df = pd.read_csv(input_path, header=None, usecols=range(16))
df.columns = columns

# === Clean & Standardize ===
df['date_of_transfer'] = pd.to_datetime(df['date_of_transfer'], errors='coerce')
df['town/city'] = df['town/city'].astype(str).str.upper()

# === Extract Year of Transaction ===
df['year_of_transaction'] = df['date_of_transfer'].dt.year

# === Define Town Groups (exclude PETERLEE) ===
core_towns = ['CORBY','BICESTER','KENILWORTH']
always_station = ['REDDITCH','KETTERING']
no_station = ['WISBECH','RUSHDEN']  # ตัด PETERLEE ออก

target_cities = core_towns + always_station + no_station

# === Filter Selected Towns only (ไม่กรองปี) ===
df_filtered = df[
    (df['town/city'].isin(target_cities))
].copy()

# === Assign Period Labels ===
def assign_period(row):
    town = row['town/city']
    date = row['date_of_transfer']
    if pd.isna(date):
        return 'Unknown'
    if town == 'CORBY':
        return 'Pre' if date < pd.Timestamp('2009-02-01') else 'Post'
    if town == 'BICESTER':
        return 'Pre' if date < pd.Timestamp('2016-12-01') else 'Post'
    if town == 'KENILWORTH':
        return 'Pre' if date < pd.Timestamp('2018-04-30') else 'Post'
    if town in always_station:
        return 'Control-Station'
    if town in no_station:
        return 'Control-NoStation'
    return 'Unknown'

df_filtered['railway_period'] = df_filtered.apply(assign_period, axis=1)

# === Save Updated Dataset ===
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_filtered.to_csv(output_path, index=False)

# === Summaries ===
print("Updated dataset saved to:", output_path)
print("\nCounts by group:")
print(df_filtered['railway_period'].value_counts())
print("\nCounts by town & period:")
print(df_filtered.groupby(['town/city','railway_period']).size())

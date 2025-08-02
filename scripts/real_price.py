import pandas as pd

# === Input and output file paths ===
property_file = "INPUT YOUR FILE PATH HERE"
inflation_file = "INPUT YOUR FILE PATH HERE"
output_file = "INPUT YOUR FILE PATH HERE"

# === Load datasets ===
df = pd.read_csv(property_file, low_memory=False)
inflation = pd.read_csv(inflation_file)

# Clean column names
inflation.columns = inflation.columns.str.strip().str.lower()

# Extract year and convert to integer
inflation['year_of_transaction'] = inflation['year'].astype(str).str.extract(r'(\d{4})')
inflation['year_of_transaction'] = pd.to_numeric(inflation['year_of_transaction'], errors='coerce')

# Keep only valid rows with CPI
inflation = inflation.dropna(subset=['year_of_transaction', 'cpi index 00: all items 2015=100'])

# Calculate annual CPI average
inflation_annual = (
    inflation
    .groupby('year_of_transaction')['cpi index 00: all items 2015=100']
    .mean()
    .reset_index()
    .rename(columns={'cpi index 00: all items 2015=100': 'cpi_annual'})
)

# Extract year from transaction date
df['year_of_transaction'] = pd.to_datetime(df['date_of_transfer'], errors='coerce').dt.year

# Merge with annual CPI
df = df.merge(inflation_annual, on='year_of_transaction', how='left')

# Set base CPI (2024)
cpi_base = inflation_annual[inflation_annual['year_of_transaction'] == 2024]['cpi_annual'].mean()

# Adjust real price
df['real_price'] = df['price'] * (cpi_base / df['cpi_annual'])

# Save final dataset
df.to_csv(output_file, index=False)
print("Final dataset saved with real_price adjusted using annual CPI.")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# === Setup ===
file_path = "INPUT YOUR FILE PATH HERE"
output_folder = "INPUT YOUR FILE PATH HERE"
os.makedirs(output_folder, exist_ok=True)

# === Load data ===
df = pd.read_csv(file_path, low_memory=False)
df = df[df['real_price'] < 2_000_000]
df = df[df['distance_to_station'] >= 0]

# === Fix dash type and remove Control-NoStation ===
df['railway_period'] = df['railway_period'].astype(str).str.strip().str.replace("–", "-")
df = df[df['railway_period'] != 'Control-NoStation']

# === Define more detailed distance bins (every 250m up to 3km, then >3km) ===
bins = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, np.inf]
labels = [
    '<250m', '250–500m', '500–750m', '750m–1km', '1–1.25km', '1.25–1.5km',
    '1.5–1.75km', '1.75–2km', '2–2.25km', '2.25–2.5km', '2.5–2.75km', '2.75–3km', '>3km'
]
df['distance_bin'] = pd.cut(df['distance_to_station'], bins=bins, labels=labels, include_lowest=True)

# === Group by railway_period and distance_bin, then compute median ===
grouped = df.groupby(['railway_period', 'distance_bin'])['real_price'].median().reset_index()

# === Plot ===
plt.figure(figsize=(14, 7))
sns.lineplot(data=grouped, x='distance_bin', y='real_price', hue='railway_period', marker='o')
plt.title("Median Inflation-adjusted Property Price by Distance to Station (Binned, Fine-grained)")
plt.xlabel("Distance to Nearest Station")
plt.ylabel("Median Inflation-adjusted Price (£)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_folder}/Figure_X_MedianInflationAdjustedPrice_by_DistanceBin_FineGrained.png")
plt.show()

print("Done: fine-grained binned plot saved & displayed with inflation-adjusted terminology.")

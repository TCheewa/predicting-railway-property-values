import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import kruskal
import os
import numpy as np

# === Setup ===
file_path = "INPUT YOUR FILE PATH HERE"
output_folder = "INPUT YOUR FILE PATH HERE"
os.makedirs(output_folder, exist_ok=True)

# === Load Data ===
df = pd.read_csv(file_path, low_memory=False)

# === Basic Cleaning ===
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['real_price'] = pd.to_numeric(df['real_price'], errors='coerce')
df['business_count'] = pd.to_numeric(df['business_count'], errors='coerce')
df['railway_period'] = df['railway_period'].fillna('Unknown')
df['year_of_transaction'] = pd.to_datetime(df['date_of_transfer'], errors='coerce').dt.year

# === Log-transform variables already included ===

sns.set(style="whitegrid")

# === 1. Histogram: Real Price (log-scale X) ===
plt.figure(figsize=(10, 5))
sns.histplot(df['real_price'][df['real_price'] > 0], bins=100, log_scale=(True, False), kde=True)
plt.title("Distribution of Inflation-adjusted Property Price (Log X)")
plt.xlabel("Inflation-adjusted Price (£, Log Scale)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{output_folder}/Figure_1_InflationPrice_Distribution_LogX.png")
plt.close()

# === 2. Histogram: Log-Transformed Price ===
plt.figure(figsize=(10, 5))
sns.histplot(df['log_price'], bins=100, kde=True)
plt.title("Distribution of Log-Transformed Inflation-adjusted Price")
plt.xlabel("Log(1 + Inflation-adjusted Price)")
plt.tight_layout()
plt.savefig(f"{output_folder}/Figure_2_LogInflationPrice_Distribution.png")
plt.close()

# === 3. Distance Distribution ===
plt.figure(figsize=(10, 5))
sns.histplot(df['distance_to_station'][df['distance_to_station'] >= 0], bins=100, kde=True)
plt.title("Distribution of Distance to Nearest Station (meters)")
plt.xlabel("Distance to Station")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{output_folder}/Figure_3_Distance_Distribution.png")
plt.close()

# === 4. Scatter: Distance vs Real Price (Log Y) ===
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df[(df['real_price'] > 0) & (df['distance_to_station'] >= 0)],
                x='distance_to_station', y='real_price', hue='railway_period', alpha=0.5)
plt.yscale('log')
plt.title("Distance to Station vs Inflation-adjusted Property Price (Log Y)")
plt.xlabel("Distance to Station (meters)")
plt.ylabel("Inflation-adjusted Price (£, Log)")
plt.tight_layout()
plt.savefig(f"{output_folder}/Figure_4_Scatter_Distance_vs_InflationPrice_LogY.png")
plt.close()

# === 5. Scatter: log_distance vs log_price + Regression ===
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='log_distance', y='log_price', alpha=0.3)
sns.regplot(data=df, x='log_distance', y='log_price', scatter=False, color='red')
plt.title("Log(Distance) vs Log(Inflation-adjusted Price)")
plt.xlabel("log(1 + Distance to Station)")
plt.ylabel("log(1 + Inflation-adjusted Price)")
plt.tight_layout()
plt.savefig(f"{output_folder}/Figure_5_LogDistance_vs_LogInflationPrice_Reg.png")
plt.close()

# === 6. Boxplot: Real Price by Railway Period ===
plt.figure(figsize=(8, 5))
sns.boxplot(data=df[df['real_price'] < 2_000_000], x='railway_period', y='real_price')
plt.title("Inflation-adjusted Property Price by Railway Period")
plt.xlabel("Railway Period")
plt.ylabel("Inflation-adjusted Price (£)")
plt.tight_layout()
plt.savefig(f"{output_folder}/Figure_6_Boxplot_InflationPrice_Period.png")
plt.close()

# === 7. Trend: Real Price by Year and Period ===
plt.figure(figsize=(10, 6))
sns.lineplot(data=df[df['real_price'] < 2_000_000],
             x='year_of_transaction', y='real_price',
             hue='railway_period', estimator='median')
plt.title("Median Inflation-adjusted Property Price by Year and Railway Period")
plt.xlabel("Year")
plt.ylabel("Median Inflation-adjusted Price (£)")
plt.tight_layout()
plt.savefig(f"{output_folder}/Figure_7_InflationPrice_Trend_Year.png")
plt.close()

# === 8. Kruskal-Wallis Test on Real Price by Railway Period ===
df_filtered = df[df['real_price'] < 2_000_000].copy()
groups = [g["real_price"].dropna() for _, g in df_filtered.groupby("railway_period")]
stat, p = kruskal(*groups)
with open(f"{output_folder}/Kruskal_InflationPrice_Result.txt", "w") as f:
    f.write("📊 Kruskal-Wallis Test on Inflation-adjusted Price:\n")
    f.write(f"  H-statistic: {stat:.4f}\n")
    f.write(f"  p-value    : {p:.4f}\n")
    f.write("  Result     : " + ("Significant difference\n" if p < 0.05 else "No significant difference\n"))

# === 9. Descriptive Table ===
desc = df.groupby('railway_period')[['real_price', 'distance_to_station', 'business_count']].agg(['mean', 'median']).round(2)
desc.to_csv(f"{output_folder}/Descriptive_Table_InflationPrice.csv")

# === 10. Optional: Facet Histogram by Railway Period ===
g = sns.FacetGrid(df[df['real_price'] < 2_000_000], col="railway_period", height=4)
g.map_dataframe(sns.histplot, x="real_price", bins=50)
g.set_titles("{col_name}")
g.set_axis_labels("Inflation-adjusted Price (£)", "Count")
plt.tight_layout()
g.savefig(f"{output_folder}/Figure_8_FacetHist_InflationPrice_PerPeriod.png")
plt.close()

# === 11. Correlation Heatmap ===
columns_for_corr = ['real_price', 'distance_to_station', 'business_count', 'year_of_transaction']
df_corr = df[columns_for_corr].copy()
for col in columns_for_corr:
    df_corr[col] = pd.to_numeric(df_corr[col], errors='coerce')
df_corr.dropna(inplace=True)
corr_matrix = df_corr.corr(method='pearson').round(2)

# Rename for display only
corr_matrix.rename(index={'real_price': 'inflation_adjusted_price'},
                   columns={'real_price': 'inflation_adjusted_price'}, inplace=True)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Heatmap (Inflation-adjusted Price)")
plt.tight_layout()
plt.savefig(f"{output_folder}/Figure_9_Correlation_Heatmap_InflationPrice.png")
plt.close()

print("All EDA plots and statistics updated to 'Inflation-adjusted Price'.")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === Setup ===
file_path = "INPUT YOUR FILE PATH HERE"
output_folder = "INPUT YOUR FILE PATH HERE"
os.makedirs(output_folder, exist_ok=True)

# === Load Data ===
df = pd.read_csv(file_path, low_memory=False)

# === Clean Columns ===
df['real_price'] = pd.to_numeric(df['price'], errors='coerce')
df['business_count'] = pd.to_numeric(df['business_count'].astype(str).str.replace(',', ''), errors='coerce')
df['railway_period'] = df['railway_period'].fillna('Unknown')
df['year_of_transaction'] = pd.to_datetime(df['date_of_transfer'], errors='coerce').dt.year
df['town/city'] = df['town/city'].astype(str).str.upper().str.strip()

# === Define Final Target Towns ===
target_towns = [
    'BICESTER', 'CORBY', 'KENILWORTH',
    'REDDITCH', 'KETTERING',
    'WISBECH', 'RUSHDEN'
]

# === Style ===
sns.set(style="whitegrid")

# === EDA per Town ===
for town in target_towns:
    subset = df[(df['town/city'] == town) & (df['real_price'] < 2_000_000)].copy()
    subset = subset[(subset['real_price'].notna()) & (subset['distance_to_station'].notna())]

    if subset.empty:
        continue

    town_folder = os.path.join(output_folder, town.replace(" ", "_"))
    os.makedirs(town_folder, exist_ok=True)

    # --- Boxplot ---
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=subset, x='railway_period', y='real_price')
    plt.title(f"{town} - Inflation-adjusted Property Price by Railway Period")
    plt.ylabel("Inflation-adjusted Price (£)")
    plt.tight_layout()
    plt.savefig(f"{town_folder}/Boxplot_InflationAdjustedPrice_Period.png")
    plt.close()

    # --- Scatterplot ---
    scatter_data = subset[(subset['real_price'] > 0) & (subset['distance_to_station'] >= 0)]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=scatter_data, x='distance_to_station', y='real_price', hue='railway_period', alpha=0.5)
    plt.yscale('log')
    plt.title(f"{town} - Distance to Station vs Inflation-adjusted Property Price")
    plt.xlabel("Distance to Station (meters)")
    plt.ylabel("Inflation-adjusted Price (£, Log Scale)")
    plt.tight_layout()
    plt.savefig(f"{town_folder}/Scatter_Distance_vs_InflationAdjustedPrice.png")
    plt.close()

    # --- Line Trend ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=subset, x='year_of_transaction', y='real_price', hue='railway_period', estimator='median')
    plt.title(f"{town} - Median Inflation-adjusted Price Trend by Year")
    plt.xlabel("Year")
    plt.ylabel("Median Inflation-adjusted Price (£)")
    plt.tight_layout()
    plt.savefig(f"{town_folder}/Line_Trend_InflationAdjustedPrice_Year.png")
    plt.close()

    # --- FacetGrid by Railway Period ---
    g = sns.FacetGrid(scatter_data, col='railway_period', height=5)
    g.map_dataframe(sns.scatterplot, x='distance_to_station', y='real_price', alpha=0.5)
    g.set_axis_labels("Distance to Station (meters)", "Inflation-adjusted Price (£, Log Scale)")
    g.set_titles(col_template="{col_name}")
    for ax in g.axes.flatten():
        ax.set_yscale('log')
    plt.tight_layout()
    g.savefig(f"{town_folder}/Facet_Distance_vs_InflationAdjustedPrice_Period.png")
    plt.close()

    # --- Correlation Heatmap (split Pre/Post if applicable) ---
    base_cols = ['real_price', 'distance_to_station', 'business_count', 'year_of_transaction']
    if town in ['BICESTER', 'CORBY', 'KENILWORTH']:
        for period in ['Pre', 'Post']:
            period_subset = subset[subset['railway_period'].str.upper() == period.upper()]
            if len(period_subset) < 10:
                continue
            corr = period_subset[base_cols].corr()
            plt.figure(figsize=(8, 6))
            ax = sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                             yticklabels=['Inflation-adjusted Price', 'Distance to Station', 'Business Count', 'Year'])
            ax.set_title(f"{town} - Correlation Heatmap ({period})")
            plt.tight_layout()
            plt.savefig(f"{town_folder}/Correlation_Heatmap_{period}_InflationAdjusted.png")
            plt.close()
    else:
        corr = subset[base_cols].corr()
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                         yticklabels=['Inflation-adjusted Price', 'Distance to Station', 'Business Count', 'Year'])
        ax.set_title(f"{town} - Correlation Heatmap (Inflation-adjusted Price)")
        plt.tight_layout()
        plt.savefig(f"{town_folder}/Correlation_Heatmap_InflationAdjustedPrice.png")
        plt.close()

print("Town-level EDA (Inflation-adjusted wording applied) complete.")

import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

# === Setup ===
file_path = "INPUT YOUR FILE PATH HERE"
output_folder = "INPUT YOUR FILE PATH HERE"
os.makedirs(output_folder, exist_ok=True)

# === Load & Clean ===
df = pd.read_csv(file_path, low_memory=False)
df['real_price'] = pd.to_numeric(df['real_price'], errors='coerce')
df['year_of_transaction'] = pd.to_datetime(df['date_of_transfer'], errors='coerce').dt.year
df['town/city'] = df['town/city'].str.upper().str.strip()
df = df[df['real_price'] < 2_000_000]
df = df[df['year_of_transaction'] >= 1995]

# === Manual railway_period mapping without Peterlee ===
manual_period_map = {
    'BICESTER': 'Pre → Post',
    'CORBY': 'Pre → Post',
    'KENILWORTH': 'Pre → Post',
    'REDDITCH': 'Control–Station',
    'KETTERING': 'Control–Station',
    'WISBECH': 'Control–NoStation',
    'RUSHDEN': 'Control–NoStation'
}
df['railway_period'] = df['town/city'].map(manual_period_map)
df = df[df['railway_period'].notna()]

# === Transition years for 'Post' period (manually defined or inferred)
transition_years = {
    'BICESTER': 2015,
    'CORBY': 2009,
    'KENILWORTH': 2018
}

# === Group Colors
group_colors = {
    'Pre → Post': 'royalblue',
    'Control–Station': 'forestgreen',
    'Control–NoStation': 'darkorange'
}

# === Line Styles per Town
town_styles = {
    'CORBY': {'linestyle': '--'},
    'KENILWORTH': {'linestyle': '-.'},
    'BICESTER': {'linestyle': ':'},
    'REDDITCH': {'linestyle': (0, (1, 2))},
    'KETTERING': {'linestyle': (0, (3, 1, 1, 1))},
    'WISBECH': {'linestyle': (0, (4, 2))},
    'RUSHDEN': {'linestyle': (0, (1, 3))}
}

# === Plot Start ===
plt.figure(figsize=(14, 7))

# === Plot Each Town ===
for town in df['town/city'].unique():
    subset = df[df['town/city'] == town]
    period = subset['railway_period'].iloc[0]
    color = group_colors[period]
    style = town_styles.get(town, {'linestyle': '-'})
    
    grouped = subset.groupby('year_of_transaction')['real_price'].median()
    
    plt.plot(grouped.index, grouped.values,
             label=f"{town.title()} ({period})",
             linestyle=style['linestyle'],
             color=color,
             linewidth=2,
             alpha=0.9)

# === Add Transition Year Lines (Only Pre → Post)
for town, year in transition_years.items():
    if town in df['town/city'].unique():
        color = group_colors.get('Pre → Post', 'gray')
        style = town_styles.get(town, {'linestyle': '--'})
        
        plt.axvline(x=year, color=color, linestyle=style['linestyle'], alpha=0.6, linewidth=1.5)
        median_price = df[(df['town/city'] == town) & (df['year_of_transaction'] == year)]['real_price'].median()
        
        if not pd.isna(median_price):
            plt.scatter(year, median_price, color=color, s=60, edgecolors='black', zorder=5)
            plt.text(
                year + 0.3,
                median_price * 1.05,
                f"{town.title()} (Post {year})",
                rotation=90,
                color=color,
                fontsize=9,
                ha='left',
                va='bottom'
            )

# === Labels and Legends ===
plt.title("Median Inflation-adjusted Property Price Trend by Town (Grouped by Railway Period)")
plt.xlabel("Year of Transaction")
plt.ylabel("Median Inflation-adjusted Price (£)")

# === Dual Legends
group_legend = [
    Line2D([0], [0], color='royalblue', lw=3, label='Pre → Post'),
    Line2D([0], [0], color='forestgreen', lw=3, label='Control–Station'),
    Line2D([0], [0], color='darkorange', lw=3, label='Control–NoStation')
]
first_legend = plt.legend(handles=group_legend, title="Railway Period (Color)", loc='upper left', bbox_to_anchor=(1.02, 1))
plt.gca().add_artist(first_legend)

plt.legend(title="Town (Line Style)", bbox_to_anchor=(1.02, 0.55), loc='upper left')

# === Save
plt.tight_layout()
plt.savefig(f"{output_folder}/Trend_Median_InflationPrice_PerTown_RailwayPeriod.png")
plt.close()

print("Multi-town trend chart with updated 'Inflation-adjusted Price' terminology saved successfully.")

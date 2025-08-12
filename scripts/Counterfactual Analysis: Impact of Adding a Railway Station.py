import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
import os

# === Load Dataset ===
df = pd.read_csv("INPUT YOUR FILE PATH HERE")

# === Features and Target ===
feature_cols = [
    'log_distance_to_station', 'business_count', 'log_interaction',
    'year_of_transaction', 'railway_Control-NoStation', 'railway_Control-Station',
    'railway_Post', 'railway_Pre'
]
target_col = 'log_real_price'

# === Train LightGBM model ===
X = df[feature_cols]
y = df[target_col]
model_lgbm = LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
model_lgbm.fit(X, y)

# === Function: Prepare Counterfactual Data ===
def prepare_counterfactual_df(town_name):
    subset = df[df['town/city'].str.upper() == town_name.upper()].copy()
    if subset.empty:
        raise ValueError(f"Town '{town_name}' not found.")
    subset_cf = subset.copy()
    subset_cf['railway_Control-NoStation'] = 0
    subset_cf['railway_Post'] = 1
    X_counter = subset_cf[feature_cols]
    log_pred = model_lgbm.predict(X_counter)
    subset['real_price'] = np.expm1(subset['log_real_price'])
    subset['predicted_price_if_has_station'] = np.expm1(log_pred)
    subset['price_diff'] = subset['predicted_price_if_has_station'] - subset['real_price']
    subset['date_of_transfer'] = pd.to_datetime(subset['date_of_transfer'])
    subset['year'] = subset['date_of_transfer'].dt.year
    return subset

# === Visualisation Functions ===
def plot_bar_real_vs_counterfactual(town_name, save_path=None):
    d = prepare_counterfactual_df(town_name)
    means = {
        'Real (No Station)': d['real_price'].mean(),
        'Post (Counterfactual)': d['predicted_price_if_has_station'].mean()
    }
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(list(means.keys()), list(means.values()))
    ax.set_title(f"{town_name.title()}: Mean Price – Real vs Counterfactual")
    ax.set_ylabel("Price (£)")
    for i, v in enumerate(means.values()):
        ax.text(i, v, f"£{v:,.0f}", ha='center', va='bottom')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.show()

def plot_diff_distribution(town_name, save_path=None):
    d = prepare_counterfactual_df(town_name)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(d['price_diff'], bins=30)
    ax.axvline(d['price_diff'].mean(), linestyle='--', color='red')
    ax.set_title(f"{town_name.title()}: Distribution of Price Uplift")
    ax.set_xlabel("Uplift (£)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.show()

def plot_monthly_timeseries(town_name, save_path=None):
    d = prepare_counterfactual_df(town_name)
    d['month'] = d['date_of_transfer'].dt.to_period('M').dt.to_timestamp()
    g = d.groupby('month').agg(real=('real_price','median'),
                               post=('predicted_price_if_has_station','median')).reset_index()
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(g['month'], g['real'], label='Real (No Station)')
    ax.plot(g['month'], g['post'], label='Post (Counterfactual)')
    ax.set_title(f"{town_name.title()}: Monthly Median Price")
    ax.set_xlabel("Month")
    ax.set_ylabel("Median Price (£)")
    ax.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.show()

def plot_uplift_by_distance_bin(town_name, n_bins=5, save_path=None):
    d = prepare_counterfactual_df(town_name)
    d['dist_bin'] = pd.qcut(d['log_distance_to_station'], q=n_bins, duplicates='drop')
    s = d.groupby('dist_bin')['price_diff'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(s['dist_bin'].astype(str), s['price_diff'])
    ax.set_title(f"{town_name.title()}: Mean Uplift by Distance Bin")
    ax.set_xlabel("Distance bin (log)")
    ax.set_ylabel("Mean Uplift (£)")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.show()

def plot_multi_town_overview(town_names, save_path=None):
    rows = []
    for t in town_names:
        try:
            d = prepare_counterfactual_df(t)
            rows.append({
                'town': t.upper(),
                'mean_real': d['real_price'].mean(),
                'mean_post': d['predicted_price_if_has_station'].mean(),
                'mean_uplift': d['price_diff'].mean()
            })
        except ValueError:
            pass
    out = pd.DataFrame(rows)
    if out.empty:
        print("No towns found.")
        return
    fig, ax = plt.subplots(figsize=(8,4))
    idx = np.arange(len(out))
    width = 0.35
    ax.bar(idx - width/2, out['mean_real'], width, label='Real')
    ax.bar(idx + width/2, out['mean_post'], width, label='Post (CF)')
    ax.set_xticks(idx)
    ax.set_xticklabels(out['town'])
    ax.set_ylabel("Mean Price (£)")
    ax.set_title("Mean Price by Town: Real vs Counterfactual")
    ax.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.show()
    print(out[['town','mean_uplift']].round(0))

# === Example Usage ===
output_dir = "INPUT YOUR FILE PATH HERE"
os.makedirs(output_dir, exist_ok=True)

# Wisbech
plot_bar_real_vs_counterfactual("WISBECH", save_path=f"{output_dir}/wisbech_bar.png")
plot_diff_distribution("WISBECH", save_path=f"{output_dir}/wisbech_hist.png")
plot_monthly_timeseries("WISBECH", save_path=f"{output_dir}/wisbech_ts.png")
plot_uplift_by_distance_bin("WISBECH", save_path=f"{output_dir}/wisbech_uplift_by_dist.png")

# Rushden
plot_bar_real_vs_counterfactual("RUSHDEN", save_path=f"{output_dir}/rushden_bar.png")
plot_diff_distribution("RUSHDEN", save_path=f"{output_dir}/rushden_hist.png")
plot_monthly_timeseries("RUSHDEN", save_path=f"{output_dir}/rushden_ts.png")
plot_uplift_by_distance_bin("RUSHDEN", save_path=f"{output_dir}/rushden_uplift_by_dist.png")

# Overview
plot_multi_town_overview(["WISBECH","RUSHDEN"], save_path=f"{output_dir}/overview_bar.png")

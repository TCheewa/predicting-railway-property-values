import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from lightgbm import LGBMRegressor

# === Paths ===
file_path = "INPUT YOUR FILE PATH HERE"
output_folder = "INPUT YOUR FILE PATH HERE"
os.makedirs(output_folder, exist_ok=True)

# === Load Data ===
df = pd.read_csv(file_path)

# === Features & Target ===
feature_cols = [
    'log_distance_to_station',
    'business_count',
    'log_interaction',
    'year_of_transaction',
    'railway_Control-NoStation',
    'railway_Control-Station',
    'railway_Post',
    'railway_Pre'
]
target_col = 'log_real_price'

X = df[feature_cols]
y = df[target_col]

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Models ===
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
}

# === Evaluation function ===
def evaluate_model(name, y_true, y_pred):
    return {
        "Model": name,
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "R²": round(r2_score(y_true, y_pred), 4)
    }

def plot_preds(y_true, y_pred, model_name):
    y_true_exp = np.expm1(y_true)
    y_pred_exp = np.expm1(y_pred)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_exp, y_pred_exp, alpha=0.3)
    plt.plot([y_true_exp.min(), y_true_exp.max()],
             [y_true_exp.min(), y_true_exp.max()], 'r--')
    plt.xlabel("Actual Price (£)")
    plt.ylabel("Predicted Price (£)")
    plt.title(f"{model_name} - Actual vs Predicted (Real Price)")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{model_name.replace(' ', '_')}_Actual_vs_Predicted.png")
    plt.close()

# === Train, Evaluate, Save Results ===
results = []
model_objects = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append(evaluate_model(name, y_test, y_pred))
    plot_preds(y_test, y_pred, name)
    model_objects[name] = model

# === Save Summary Table ===
results_df = pd.DataFrame(results)
results_df.to_csv(f"{output_folder}/model_performance_summary.csv", index=False)

# === XGBoost Feature Importance ===
importances = model_objects["XGBoost"].feature_importances_
feature_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance')

plt.figure(figsize=(8, 6))
plt.barh(feature_df['Feature'], feature_df['Importance'], color='steelblue')
plt.title("Feature Importance from XGBoost")
plt.tight_layout()
plt.savefig(f"{output_folder}/XGBoost_Feature_Importance.png")
plt.close()

# === Counterfactual Function ===
def simulate_counterfactual(town_name):
    subset = df[df['town/city'].str.upper() == town_name.upper()].copy()
    if subset.empty:
        print(f"Town '{town_name}' not found in dataset.")
        return
    subset['railway_Control-NoStation'] = 0
    subset['railway_Post'] = 1

    X_counter = subset[feature_cols]
    log_pred = model_objects["XGBoost"].predict(X_counter)
    subset['predicted_price_if_has_station'] = np.expm1(log_pred)
    subset['real_price'] = np.expm1(subset['log_real_price'])
    subset['price_diff'] = subset['predicted_price_if_has_station'] - subset['real_price']

    output_file = f"{output_folder}/{town_name.upper()}_Counterfactual.csv"
    cols_to_save = ['postcode', 'date_of_transfer', 'real_price', 'predicted_price_if_has_station', 'price_diff']
    available_cols = [col for col in cols_to_save if col in subset.columns]
    subset[available_cols].to_csv(output_file, index=False)
    print(f"Counterfactual for {town_name} saved to {output_file}")

# === Run Counterfactuals ===
simulate_counterfactual("WISBECH")
simulate_counterfactual("RUSHDEN")

print("\n All models completed and counterfactuals saved.")

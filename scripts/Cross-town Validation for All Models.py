import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

# === SETUP ===
file_path = "INPUT YOUR FILE PATH HERE"
output_dir = "INPUT YOUR FILE PATH HERE"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(file_path)
feature_cols = [
    'log_distance_to_station', 'business_count', 'log_interaction',
    'year_of_transaction', 'railway_Control-NoStation', 'railway_Control-Station',
    'railway_Post', 'railway_Pre'
]
target_col = 'log_real_price'

towns = df['town/city'].dropna().unique()
results = []

# === FUNCTION: Evaluate ===
def evaluate(y_true, y_pred):
    return {
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "R²": round(r2_score(y_true, y_pred), 4)
    }

# === LOOP OVER TOWNS ===
for test_town in towns:
    train_df = df[df['town/city'] != test_town]
    test_df = df[df['town/city'] == test_town]

    if len(test_df) < 100:
        print(f"Skipping {test_town}: Not enough data ({len(test_df)} rows)")
        continue

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # === Model 1: Linear Regression ===
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    res_lr = evaluate(y_test, y_pred_lr)
    res_lr.update({"Model": "Linear Regression", "Town": test_town})

    # === Model 2: Random Forest ===
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    res_rf = evaluate(y_test, y_pred_rf)
    res_rf.update({"Model": "Random Forest", "Town": test_town})

    # === Model 3: XGBoost ===
    xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    res_xgb = evaluate(y_test, y_pred_xgb)
    res_xgb.update({"Model": "XGBoost", "Town": test_town})

    # === Model 4: LightGBM ===
    lgbm_model = LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
    lgbm_model.fit(X_train, y_train)
    y_pred_lgbm = lgbm_model.predict(X_test)
    res_lgbm = evaluate(y_test, y_pred_lgbm)
    res_lgbm.update({"Model": "LightGBM", "Town": test_town})

    # === Append all results ===
    results.extend([res_lr, res_rf, res_xgb, res_lgbm])

    # === Optional: Save chart for each model (log price) ===
    for model_name, y_pred in zip(
        ["Linear_Regression", "Random_Forest", "XGBoost", "LightGBM"],
        [y_pred_lr, y_pred_rf, y_pred_xgb, y_pred_lgbm]
    ):
        plt.figure(figsize=(5.5, 5.5))
        plt.scatter(np.expm1(y_test), np.expm1(y_pred), alpha=0.3)
        plt.plot([np.expm1(y_test).min(), np.expm1(y_test).max()],
                 [np.expm1(y_test).min(), np.expm1(y_test).max()], 'r--')
        plt.xlabel("Actual Price (£)")
        plt.ylabel("Predicted Price (£)")
        plt.title(f"{model_name} on {test_town}")
        plt.tight_layout()
        chart_path = f"{output_dir}/{test_town}_{model_name}_Pred_vs_Actual.png"
        plt.savefig(chart_path)
        plt.close()

    print(f"Finished: {test_town}")

# === Save Results ===
results_df = pd.DataFrame(results)
results_df = results_df[["Town", "Model", "RMSE", "MAE", "R²"]]
results_df.to_csv(f"{output_dir}/cross_town_all_models_with_lgbm.csv", index=False)

print(f"\n All model results saved to {output_dir}/cross_town_all_models_with_lgbm.csv")

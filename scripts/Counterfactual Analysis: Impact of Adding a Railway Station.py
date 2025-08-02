import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === File Paths ===
wisbech_path = "INPUT YOUR FILE PATH HERE"
rushden_path = "INPUT YOUR FILE PATH HERE"
output_path = "INPUT YOUR FILE PATH HERE"

# === Load Data ===
df_w = pd.read_csv(wisbech_path)
df_r = pd.read_csv(rushden_path)

# === Add Town Labels ===
df_w["town"] = "WISBECH"
df_r["town"] = "RUSHDEN"

# === Combine ===
df_all = pd.concat([df_w, df_r], ignore_index=True)

# === Summary Table ===
summary = df_all.groupby("town")[["real_price", "predicted_price_if_has_station", "price_diff"]].mean().round(2)
summary["% increase"] = 100 * (summary["predicted_price_if_has_station"] - summary["real_price"]) / summary["real_price"]
print("\n Average Counterfactual Price Difference:")
print(summary)

# === Plot Comparison ===
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_all, x="town", y="price_diff", palette="Set2")
plt.title("Predicted Price Difference if Railway Station is Added")
plt.ylabel("Price Difference (£)")
plt.xlabel("Town")
plt.tight_layout()
plt.savefig(output_path)
plt.show()

print(f"\n Chart saved to: {output_path}")

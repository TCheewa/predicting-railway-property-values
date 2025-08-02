# Railway Impact on Property Prices – Predictive Modeling Pipeline

This repository contains a full end-to-end pipeline for analyzing and predicting property prices and business activity in railway-access versus railway-free towns using machine learning. It is part of a master's dissertation project exploring how railway infrastructure investments can affect real estate values and local business development.

---

## Folder: `/script/`

This folder contains the ordered Python scripts used to clean, merge, transform, and model the data.

> **Important:** All file paths inside the scripts are marked with placeholder text like:
```
INPUT YOUR FILE PATH HERE
```
Update these paths before running the scripts.

---

## Pipeline Overview

The pipeline follows this logical sequence:

### 1. **Data Cleaning**
| Script | Purpose | Input | Output |
|--------|---------|--------|--------|
| `clean_price_data.py` | Clean raw UK Land Registry price-paid data and assign railway access group | Raw CSV from [Land Registry](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads) | Cleaned price-paid data for target towns |
| `clean_naptan_data.py` | Filter active railway stations and extract creation year | Filtered NaPTAN CSV | List of stations with year and location |
| `merge_price_nomis.py` | Merge business counts from ONS/Nomis with transaction data by district and year | Price data + Nomis business data | Enriched dataset with business counts |
| `merge_codepoint_latlon.py` | Merge postcode coordinates from Ordnance Survey Code-Point Open | Code-Point Open CSVs + property data | Add Latitude/Longitude to property records |

---

### 2. **Feature Engineering**
| Script | Purpose | Input | Output |
|--------|---------|--------|--------|
| `distance_merge_price.py` | Calculate time-aware distance to nearest station (based on creation year) | Price data + station list | Property dataset with distance in meters |
| `real_price.py` | Adjust prices for inflation using CPI to compute real prices | CPI data + price data | Dataset with `real_price` column |
| `postcode_with_nearest_station.py` | Match each postcode to nearest station (using BallTree search) | CodePoint + station data | Dataset mapping postcodes to nearest stations |
| `real_price_with_station_info.py` | Merge nearest station info into main dataset | Price + nearest station data | Dataset with `distance_to_station` |
| `create_interaction_features.py` | Create interaction features and indicator variables (e.g. log(distance × business)) | Final station-adjusted data | Feature-enriched dataset |
| `Outliner_cleaned.py` | Remove or winsorize outliers using IQR and transformation | Interaction dataset | Final cleaned dataset with log-transforms |

---

### 3. **Exploratory Data Analysis (EDA)**
| Script | Purpose | Output |
|--------|---------|--------|
| `eda_analysis.py` | Generate histograms, scatter plots, heatmaps, and trendlines across groups | PNGs + CSV summary |
| `eda_by_town.py` | EDA split by town for boxplots, trends, correlation | Town-specific visualizations |
| `Multi-Town_Trend Plot.py` | Visualize price trends across towns with transition markers | Single combined trend plot |
| `Binned Price vs Distance to Station_all_town.py` | Show binned price trends by distance interval | Fine-grained distance-price chart |

---

### 4. **Modeling and Validation**
| Script | Purpose | Output |
|--------|---------|--------|
| `Feature Engineering.py` | Finalize input features for ML models | Model-ready dataset |
| `Regression Modeling Code (Baseline + Tree Model+Xgboost+LGBM).py` | Train and evaluate 4 ML models | Metrics + feature importance + charts |
| `Cross-Validation 4 models.py` | Perform 5-fold CV for all models | Cross-validation metrics |
| `Cross-town Validation for All Models.py` | Cross-validation using leave-one-town-out | Metrics per town and model |

---

### 5. **Counterfactual Analysis**
| Script | Purpose | Output |
|--------|---------|--------|
| `Counterfactual Analysis: Impact of Adding a Railway Station.py` | Simulate price uplift for Wisbech and Rushden if station were added | Summary stats + boxplot chart |

---

### 6. **Geospatial Visualization**
| Script | Purpose | Output |
|--------|---------|--------|
| `map_interactive.py` | Generate interactive time-based map using Folium | `HTML` map showing price changes by town, group, station proximity |

---

## Dataset Sources

Data used in this project are from official and open-access sources. You must download these separately:

| Dataset | Link |
|---------|------|
| UK Price Paid Data | https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads |
| UK Business Counts | https://www.nomisweb.co.uk/datasets/idbrent |
| NaPTAN (railway stops) | https://beta-naptan.dft.gov.uk/download |
| Ordnance Survey Code-Point Open | https://osdatahub.os.uk/downloads/open/CodePointOpen |
| CPI Inflation Rates | https://www.ons.gov.uk/economy/inflationandpriceindices |


## ✅ License

Open for educational use. Please cite if reused for research or teaching purposes.

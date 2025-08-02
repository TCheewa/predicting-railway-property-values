# Predicting Property Prices and Business Activity in Railway-Free Towns Using Machine Learning

This repository contains the source code, model implementation, and analysis for an MSc Dissertation submitted to the University of Sheffield (2025). The project investigates the impact of railway infrastructure on property values and local business activity in small- to medium-sized towns in England, using machine learning models.

## Project Summary

Railway infrastructure plays a critical role in shaping land values and regional economic development. This research applies predictive modeling—using Random Forest, XGBoost, and LightGBM—to estimate the counterfactual uplift in property prices and business activity if railway stations were introduced in currently unserved towns.

Key features include:
- **Inflation-adjusted property prices (1995–2025)**
- **Distance to nearest station (time-aware)**
- **Local economic activity (business count per year)**
- **Interaction features such as log(distance × business_count)**
- **Cross-town validation to test model generalisability**

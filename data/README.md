This directory contains all input datasets and processed data used for the dissertation titled:

Predicting Property Values and Business Activity in Railway-Free Areas Using Machine Learning

data/
├── raw/                     # Manually downloaded raw data from official sources
├── processed/               # Cleaned, filtered, and merged datasets used in analysis

---

## Required Manual Downloads (Place in `data/raw/`)

To comply with licensing policies, the following datasets must be manually downloaded and placed in the `data/raw/` folder:

| Dataset                                | Description                                                  | Download Link |
|----------------------------------------|--------------------------------------------------------------|----------------|
| **UK Land Registry Price Paid Data**   | Residential property transactions (England & Wales)          | https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads |
| **UK Business Counts (Nomis)**         | Annual active businesses by local authority                  | https://www.nomisweb.co.uk/datasets/idbrent |
| **NaPTAN Dataset**                     | Locations of UK rail and bus stops                           | https://beta-naptan.dft.gov.uk/download |
| **Ordnance Survey Code-Point Open**    | UK postcode geolocation data                                 | https://osdatahub.os.uk/downloads/open/CodePointOpen |
| **ONS Consumer Price Index (CPI)**     | Monthly inflation data for adjusting property prices         | https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/d7bt/mm23 |

Please ensure that raw files are named consistently and placed into subdirectories as required by the scripts (see `/scripts/preprocessing/` for details).

---

## Processed Datasets (`data/processed/`)

These are cleaned, engineered datasets used in the project’s analysis, visualisation, and machine learning models:

| File Name                             | Description                                                         |
|--------------------------------------|---------------------------------------------------------------------|
| `cleaned_outlier_processed.csv`      | Cleaned transactions with outlier handling and CPI-adjusted price   |
| `final_enriched_with_features.csv`   | Includes distance to station, business count, CPI, etc.             |
| `final_with_interaction.csv`         | Includes interaction terms such as log(distance × business_count)   |
| `rail_stations_with_year.csv`        | Filtered station data with historical opening year tracking         |

---

## Notes and Limitations

- **Corby**: Business count data **prior to 2010** is unavailable. These years default to 0 but may **underestimate** actual economic activity before station reopening.
- **2025**: Business data is **not available** from Nomis and may appear as missing or zero for that year.
- **Postcode Matching**: Distance to station and business count were merged using local authority and postcode-level matching.


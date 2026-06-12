# NariSafe — Women Safety Risk Awareness

**Live Demo:** https://narisafe-risk-awareness.streamlit.app

NariSafe is a public-data-based women safety risk-awareness prototype for Indian cities.

It combines **NCRB crime statistics** (2021–2023), **OpenStreetMap urban infrastructure** data and engineered contextual features to classify a given situation —> city, time, area type, complaint type —> into a risk-awareness level:
- `low`
- `medium`
- `high`

The final model is a **no-location HistGradientBoostingClassifier with high-risk threshold tuning**.

The project now has two ways to run the demo:

- **Streamlit app** (`app.py`) — hosted demo version. It downloads the model and dataset directly from Hugging Face at runtime.
- **FastAPI + HTML frontend** (`backend/main.py` + `frontend/index.html`) — local/API version. It expects the model and lookup CSV to exist in the project folders.

---

> **Disclaimer**
> This is a public-data-based risk-awareness prototype using proxy labels, not guaranteed real-world crime prediction. Risk scores are derived from NCRB 2021–2023 statistics and OpenStreetMap infrastructure data using rule-based labels.

---

## Table of Contents

- [What the project does](#what-the-project-does)
- [Data sources](#data-sources)
- [Download the dataset from Hugging Face](#download-the-dataset-from-hugging-face)
- [Project structure](#project-structure)
- [How the dataset was built](#how-the-dataset-was-built)
- [Features](#features)
- [How labels were created](#how-labels-were-created)
- [Model training](#model-training)
- [Threshold tuning](#threshold-tuning)
- [Final model](#final-model)
- [Streamlit app](#streamlit-app)
- [Backend API](#backend-api)
- [Frontend](#frontend)
- [Installation](#installation)
- [Running the app](#running-the-app)
- [API reference](#api-reference)
- [What is tracked in git](#what-is-tracked-in-git)
- [Limitations](#limitations)

---

## What the project does

```
User picks:
  city, day of week, hour, area type, complaint type

         ↓

Backend looks up the pre-built feature row from the CSV
(city-level crime stats + OSM infrastructure + time + area features)

         ↓

No-location model predicts:
  P(low), P(medium), P(high)

  if P(high) >= 0.2  →  risk_awareness_level = "high"
  else               →  whichever of low/medium has higher probability

         ↓

Response returns:
  risk_awareness_level, model_confidence, key_risk_factors,
  mitigating_factors, data_limitations, context, disclaimer
```

---

## Data sources

### 1. NCRB PDF reports (raw data)

Two PDFs placed manually in `data/raw/`:

| File | What it contains |
|---|---|
| `crime_women_citywise_2021_2023.pdf` | Table 3B.1 — total women crimes per city, 2021–2023 |
| `crime_women_headwise_citywise_2023.pdf` | Crime type breakdown per city for 2023 (head-wise) |

These are official NCRB (National Crime Records Bureau) publications, publicly available on the NCRB website.

The PDFs were parsed with `pdfplumber` to extract structured city-level and crime-type-level statistics.

### 2. OpenStreetMap India PBF

File: `data/raw/india-latest.osm.pbf`

Downloaded from [Geofabrik](https://download.geofabrik.de/asia/india.html).

This file is a full dump of all OpenStreetMap data for India. It was used to extract per-city infrastructure counts within defined radii:
- police stations (5 km radius)
- public transport stops: bus stops, bus stations, railway stations (3 km radius)
- street lights (2 km radius)
- road network length and density (5 km radius)
- land use: commercial, residential, industrial, retail (5 km radius)
- education POIs (5 km radius)

Extraction was done with `pyrosm` and `geopandas`, using city center coordinates from `city_coordinates.csv`.

> The PBF file is large (~1.6 GB). It is not tracked in git. Download it yourself from Geofabrik and place it at `data/raw/india-latest.osm.pbf`.

### 3. City coordinates

File: `data/raw/city_coordinates.csv`

Hand-verified city center coordinates for all 34 cities. Used as the anchor point for OSM radius queries.

**34 cities covered:**

```
Agra, Amritsar, Asansol, Aurangabad, Bhopal, Chandigarh City,
Dhanbad, Durg-Bhilainagar, Faridabad, Gwalior, Jabalpur, Jamshedpur,
Jodhpur, Kannur, Kollam, Kota, Ludhiana, Madurai, Malappuram, Meerut,
Nasik, Prayagraj, Raipur, Rajkot, Ranchi, Srinagar,
Thiruvananthapuram, Thrissur, Tiruchirapalli, Vadodara, Varanasi,
Vasai Virar, Vijayawada, Vishakhapatnam
```

---

## Download the dataset from Hugging Face

The dataset and model are published on Hugging Face. You do not need to re-run the full pipeline if you just want to use the trained model or the pre-built dataset.

The Streamlit app (`app.py`) downloads these artifacts automatically at runtime.
The FastAPI backend (`backend/main.py`) expects the same artifacts to be present
as local files inside `data/processed/...` and `models/...`.

### Dataset repo

```
https://huggingface.co/datasets/avnisinghal001/narisafe-risk-awareness-dataset
```

**Files in the dataset repo:**

| File | Description |
|---|---|
| `narisafe_ml_features.csv` | Full feature dataset (489,804 rows, 40 columns). Includes city, latitude, longitude. Used for API lookup. |
| `narisafe_ml_features_no_location.csv` | No-location version used to train the final model. City, latitude, longitude removed. |
| `feature_config.json` | Feature column list for the full dataset |
| `feature_config_no_location.json` | Feature column list for the no-location model |
| `high_threshold_tuning.csv` | Threshold sweep results |

**Download with the Hugging Face Python library:**

```python
from huggingface_hub import hf_hub_download

# Download full feature CSV
hf_hub_download(
    repo_id="avnisinghal001/narisafe-risk-awareness-dataset",
    filename="narisafe_ml_features.csv",
    repo_type="dataset",
    local_dir="data/processed/ml_ready/",
)

# Download no-location feature CSV
hf_hub_download(
    repo_id="avnisinghal001/narisafe-risk-awareness-dataset",
    filename="narisafe_ml_features_no_location.csv",
    repo_type="dataset",
    local_dir="data/processed/ml_ready_no_location/",
)
```

**Or with the Hugging Face CLI:**

```bash
# Install the CLI
pip install huggingface_hub

# Download entire dataset repo
huggingface-cli download avnisinghal001/narisafe-risk-awareness-dataset \
    --repo-type dataset \
    --local-dir hf_dataset/
```

### Model repo

```
https://huggingface.co/avnisinghal001/narisafe-risk-awareness-model
```

**Files in the model repo:**

| File | Description |
|---|---|
| `best_no_location_threshold_model.joblib` | Final trained model bundle (pipeline + threshold + class labels) |
| `feature_config_no_location.json` | Feature config used by the model |
| `inference_example.py` | Standalone inference script |
| `baseline_model_report.md` | Full evaluation report |
| `high_threshold_tuning.csv` | Threshold tuning results |

**Download the model:**

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="avnisinghal001/narisafe-risk-awareness-model",
    filename="best_no_location_threshold_model.joblib",
    local_dir="models/no_location_baselines/",
)
```

---

## Project structure

```
NariSafe/
│
├── app.py                             Streamlit app — hosted demo, downloads HF artifacts at runtime
│
├── backend/
│   └── main.py                        FastAPI app — model loading, prediction, routes
│
├── data/
│   ├── raw/                           Not tracked in git
│   │   ├── crime_women_citywise_2021_2023.pdf
│   │   ├── crime_women_headwise_citywise_2023.pdf
│   │   ├── india-latest.osm.pbf
│   │   └── city_coordinates.csv
│   │
│   ├── intermediate/                  Not tracked in git
│   │   ├── osm_feature_summary.csv
│   │   ├── women_crime_citywise.csv
│   │   ├── women_crime_headwise_all_types.csv
│   │   └── women_crime_headwise_model_ready.csv
│   │
│   └── processed/
│       ├── ml_ready/
│       │   ├── feature_config.json            Tracked
│       │   └── narisafe_ml_features.csv       NOT tracked (download from HF)
│       └── ml_ready_no_location/
│           ├── feature_config_no_location.json  Tracked
│           └── narisafe_ml_features_no_location.csv  NOT tracked (download from HF)
│
├── frontend/
│   └── index.html                     Single-page UI
│
├── models/
│   ├── baselines/                     NOT tracked in git (joblib files)
│   └── no_location_baselines/
│       └── best_no_location_threshold_model.joblib   NOT tracked (download from HF)
│
├── reports/
│   ├── final/
│   │   ├── final_ml_summary.md
│   │   └── final_model_manifest.json
│   ├── ml/
│   │   └── feature_importance.csv     Tracked
│   └── ml_no_location/
│       ├── baseline_model_report.md   Tracked
│       └── high_threshold_tuning.csv  Tracked
│
├── scripts/
│   ├── 01_check_osm_setup.py
│   ├── 02_extract_osm_features.py
│   ├── 03_extract_citywise_women_crime.py
│   ├── 04_extract_all_headwise_women_crime.py
│   ├── 05_clean_headwise_for_model.py
│   ├── 06_generate_final_dataset.py
│   ├── 07_create_model_training_dataset.py
│   ├── 08_audit_training_dataset.py
│   ├── 09_prepare_ml_features.py
│   ├── 10_create_train_test_splits.py
│   ├── 11_train_baseline_models.py
│   ├── 12_feature_importance.py
│   ├── 13_predict_risk.py
│   ├── 14_create_no_location_features.py
│   ├── 15_create_no_location_splits.py
│   ├── 16_train_no_location_baselines.py
│   ├── 17_tune_high_risk_threshold.py
│   ├── 18_predict_threshold_risk.py
│   └── 19_create_final_ml_summary.py
│
├── hf_upload/                         NOT tracked in git (upload staging)
├── requirements.txt
└── .gitignore
```

---

## How the dataset was built

The full pipeline runs sequentially, script by script. Here is every step.

```
┌─────────────────────────────────────────────────────────┐
│                    RAW DATA SOURCES                     │
│  NCRB PDFs (city-level + head-wise)  │  OSM India PBF  │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │  Scripts 01–05          │
        │  Parse PDFs, extract    │
        │  OSM features, clean    │
        │  and normalize crime    │
        │  type names             │
        └────────────┬────────────┘
                     │ data/intermediate/
                     │   osm_feature_summary.csv
                     │   women_crime_citywise.csv
                     │   women_crime_headwise_model_ready.csv
        ┌────────────▼────────────┐
        │  Script 06              │
        │  Cross-join city × day  │
        │  × hour × area ×        │
        │  complaint_type         │
        │  Attach OSM + crime     │
        │  features               │
        │  Compute risk_score     │
        │  Assign risk_level      │
        │  (proxy label)          │
        └────────────┬────────────┘
                     │ 489,804 rows
        ┌────────────▼────────────┐
        │  Script 07              │
        │  Drop leaky columns     │
        │  (risk_score,           │
        │  label_source)          │
        │  Final ML training CSV  │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Script 08              │
        │  Audit: missing values, │
        │  distributions, class   │
        │  balance check          │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Scripts 09–10          │
        │  ML feature prep        │
        │  Train/val/test splits  │
        │  (random + group-city)  │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Scripts 11–12          │
        │  Train 4 baseline       │
        │  models on full         │
        │  (with-location)        │
        │  feature set            │
        │  Feature importance     │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Scripts 14–16          │
        │  Remove city/lat/lon    │
        │  New no-location        │
        │  splits                 │
        │  Re-train 4 models      │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Script 17              │
        │  Tune high-risk         │
        │  threshold on           │
        │  validation set         │
        │  Select threshold=0.2   │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Scripts 18–19          │
        │  Final inference test   │
        │  Summary and manifest   │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  backend/main.py        │
        │  FastAPI API + frontend │
        └─────────────────────────┘
```

### What each script does

| Script | What it does |
|---|---|
| `01_check_osm_setup.py` | Verifies required files (OSM PBF, city CSV) and packages are present before starting |
| `02_extract_osm_features.py` | Reads `india-latest.osm.pbf` with `pyrosm`, queries around each city center, counts police stations, transport stops, street lights, roads, land use, education POIs. Saves to `data/intermediate/osm_feature_summary.csv` |
| `03_extract_citywise_women_crime.py` | Parses `crime_women_citywise_2021_2023.pdf` (Table 3B.1) with `pdfplumber`. Extracts total women crime counts for each city for 2021, 2022, 2023. Saves to `data/intermediate/women_crime_citywise.csv` |
| `04_extract_all_headwise_women_crime.py` | Parses `crime_women_headwise_citywise_2023.pdf`. Extracts crime counts for raw NCRB crime-head categories (54 categories before subtotal cleanup). Saves to `data/intermediate/women_crime_headwise_all_types.csv` |
| `05_clean_headwise_for_model.py` | Normalizes crime type names to `snake_case`, removes total/subtotal rows, and keeps 42 model-ready atomic complaint types. Saves to `data/intermediate/women_crime_headwise_model_ready.csv` |
| `06_generate_final_dataset.py` | **Core dataset generation.** Cross-joins 34 cities × 7 days × 7 hours × 7 area contexts × 42 complaint types = 489,804 rows. Attaches OSM features, crime stats, time features, crowd density, engineered scores. Computes `risk_score` and assigns `risk_level` proxy label |
| `07_create_model_training_dataset.py` | Drops `risk_score` (to prevent target leakage), `complaint_type_original`, `label_source`. Saves the clean ML training CSV |
| `08_audit_training_dataset.py` | Checks for missing values, class balance, numeric distributions. Saves audit reports to `data/processed/audit/` |
| `09_prepare_ml_features.py` | Writes `feature_config.json` defining which columns are numeric vs categorical. Saves `narisafe_ml_features.csv` |
| `10_create_train_test_splits.py` | Creates two types of splits — **random** (rows shuffled) and **group-city** (entire cities held out for val/test). Saves all six CSVs |
| `11_train_baseline_models.py` | Trains 4 models (DummyClassifier, Logistic Regression, Random Forest, HistGradientBoosting) on both split types. Saves all model `.joblib` files. Saves `reports/ml/baseline_model_report.md` |
| `12_feature_importance.py` | Extracts feature importances from the Random Forest. Saves `reports/ml/feature_importance.csv` and a markdown report |
| `13_predict_risk.py` | Standalone prediction script (with-location model, for reference) |
| `14_create_no_location_features.py` | Drops `city`, `latitude`, `longitude` from the feature set. Saves `narisafe_ml_features_no_location.csv` and `feature_config_no_location.json` |
| `15_create_no_location_splits.py` | Creates the same random + group-city splits for the no-location dataset |
| `16_train_no_location_baselines.py` | Trains the same 4 models on the no-location feature set |
| `17_tune_high_risk_threshold.py` | Sweeps P(high) threshold from 0.2 to 0.7 on the validation set. Selects the threshold that maximises macro-F1 while increasing high-risk recall vs the 0.5 baseline. Saves the final bundle (pipeline + threshold + class labels) to `models/no_location_baselines/best_no_location_threshold_model.joblib` |
| `18_predict_threshold_risk.py` | Standalone inference test for the threshold model |
| `19_create_final_ml_summary.py` | Writes `reports/final/final_ml_summary.md` and `final_model_manifest.json` |

---

## Features

The final dataset has **489,804 rows** and **40 columns** (39 features + 1 target).

### Dimensions used to generate rows

Each row is a unique combination of:

| Dimension | Values |
|---|---|
| City | 34 Indian cities |
| Day of week | Monday – Sunday (7) |
| Hour | 6, 9, 12, 15, 18, 21, 23 (7 time slots) |
| Area context | residential, commercial, market, industrial, transit, educational, mixed (7) |
| Complaint type | 42 atomic NCRB women crime types |

`34 × 7 × 7 × 7 × 42 = 489,804`

### Feature groups

**Time features**

| Feature | Description |
|---|---|
| `hour` | Hour of day (6–23) |
| `is_weekend` | 1 if Saturday or Sunday |
| `time_bucket` | morning / afternoon / evening / late_night |
| `day_of_week` | Monday – Sunday |

**Complaint / crime type features**

| Feature | Description |
|---|---|
| `complaint_type_clean` | Normalized snake_case complaint type name |
| `complaint_severity` | 1 (low) / 2 (medium) / 3 (high) — assigned by crime type |
| `complaint_type_count` | Number of cases in the city for this crime type (2023 NCRB) |
| `complaint_type_share` | This type's share of total city women crimes |

**City-level crime statistics**

| Feature | Description |
|---|---|
| `women_crime_2021` | Total women crimes in city (2021) |
| `women_crime_2022` | Total women crimes in city (2022) |
| `women_crime_2023` | Total women crimes in city (2023) |
| `women_crime_rate_2023` | Crimes per lakh population (2023) |
| `women_crime_growth_21_23` | Fractional change from 2021 to 2023 |
| `population_lakhs` | City population estimate |
| `chargesheeting_rate_2023` | % of cases that resulted in a chargesheet |

**OSM infrastructure features**

| Feature | Description |
|---|---|
| `police_station_count_5km` | Police stations within 5 km of city center |
| `nearest_police_station_km` | Distance to nearest police station |
| `public_transport_count_3km` | All public transport stops within 3 km |
| `bus_stop_count_3km` | Bus stops within 3 km |
| `bus_station_count_3km` | Bus stations within 3 km |
| `railway_station_count_3km` | Railway stations within 3 km |
| `street_light_count_2km` | Street lights within 2 km |
| `road_length_km_5km` | Total road length within 5 km |
| `road_density_km_per_sqkm_5km` | Road density (km per sq km) |
| `commercial_landuse_count_5km` | Commercial land use polygons |
| `residential_landuse_count_5km` | Residential land use polygons |
| `industrial_landuse_count_5km` | Industrial land use polygons |
| `retail_landuse_count_5km` | Retail land use polygons |
| `education_poi_count_5km` | Schools, colleges, universities |
| `lighting_data_available` | 1 if OSM has any street light data for this city |

**Engineered contextual scores**

| Feature | Description |
|---|---|
| `area_context` | Area type (categorical) |
| `crowd_density` | low / medium / high — derived from area context + time |
| `lighting_score` | 1–5 — derived from street light count and lighting availability |
| `police_access_score` | 0–3 — derived from station count and distance |
| `transport_access_score` | 0–3 — derived from public transport count |
| `urban_density_score` | 0–4 — composite of road density + land use counts |

**Target**

| Column | Values |
|---|---|
| `risk_level` | `low` / `medium` / `high` |

### Class distribution in the full dataset

| Class | Count | Share |
|---|---:|---:|
| medium | 287,812 | 58.8% |
| low | 177,702 | 36.3% |
| high | 24,290 | 5.0% |

### Features removed for the no-location model

`city`, `latitude`, `longitude` dropped to prevent the model from memorizing city identity and to test whether it can learn general contextual safety patterns.

---

## How labels were created

There is no real incident-level ground truth. The `risk_level` label is a **rule-based proxy label** engineered from a weighted combination of available features.

A `risk_score` was computed for each row using:

- city women crime rate (weighted)
- complaint severity (weighted)
- complaint type share in the city
- time bucket (late night > evening > afternoon > morning)
- crowd density (low crowd = higher risk)
- lighting score
- police access score
- transport access score
- crime growth trend

`risk_score` was then converted into `risk_level`:
- Bottom tercile → `low`
- Middle tercile → `medium`
- Top tercile → `high`

`risk_score` is **never used as a model input feature**, it was dropped in script 07 to prevent target leakage.

---

## Model training

### Two evaluation approaches

| Split type | How it works | What it tests |
|---|---|---|
| **Random split** | Rows shuffled randomly 64/16/20 | Interpolation (easy, not realistic) |
| **Group-city split** | Entire cities held out for val/test | Generalization to unseen cities (realistic) |

The **group-city split is the primary benchmark** because it tests the model on cities not seen during training.

Group-city split cities:
- **Train (21 cities):** Amritsar, Asansol, Aurangabad, Bhopal, Chandigarh City, Dhanbad, Durg-Bhilainagar, Jamshedpur, Kannur, Ludhiana, Madurai, Malappuram, Nasik, Raipur, Rajkot, Srinagar, Vadodara, Varanasi, Vasai Virar, Vijayawada, Vishakhapatnam
- **Validation (6 cities):** Agra, Gwalior, Jabalpur, Jodhpur, Kollam, Tiruchirapalli
- **Test (7 cities):** Faridabad, Kota, Meerut, Prayagraj, Ranchi, Thiruvananthapuram, Thrissur

### Models compared

Four models were trained on both the full (with-location) and no-location feature sets, under both splits.

**No-location, group-city split (the honest benchmark):**

| Model | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|---:|
| HistGradientBoosting | 0.830 | 0.750 | **0.882** | **0.805** |
| Random Forest | 0.783 | 0.713 | 0.848 | 0.792 |
| Logistic Regression | 0.634 | 0.571 | 0.671 | 0.648 |
| Dummy (most frequent) | 0.671 | 0.268 | 0.638 | 0.260 |

The random split models (with-location) showed near-perfect accuracy — a sign of overfitting to city identity, not generalization.

**Why macro F1 matters more than accuracy:** the `high` risk class is only ~5% of rows. A model that predicts `medium` for everything gets 58% accuracy but learns nothing useful. Macro F1 gives equal weight to all three classes.

---

## Threshold tuning

The default `predict` call at threshold 0.5 had low recall for the `high` class — it missed most high-risk situations. Since a safety tool should be sensitive to high-risk, we tuned the threshold.

**Strategy:** sweep P(high) thresholds from 0.2 to 0.7 on the validation set. Select the threshold that maximises macro-F1 while improving high-risk recall vs the 0.5 baseline.

| Threshold | Accuracy | Macro F1 | High Precision | High Recall | High F1 |
|---:|---:|---:|---:|---:|---:|
| **0.20** | 0.8338 | **0.7668** | 0.9366 | **0.4947** | **0.6474** |
| 0.25 | 0.8333 | 0.7648 | 0.9358 | 0.4885 | 0.6419 |
| 0.30 | 0.8324 | 0.7620 | 0.9347 | 0.4797 | 0.6340 |
| 0.50 | 0.8296 | 0.7502 | 0.9486 | 0.4392 | 0.6004 |

**Selected threshold: 0.2** — best macro-F1 with highest high-risk recall.

### What the threshold does

```python
proba = pipeline.predict_proba(X)[0] 

if P(high) >= 0.2:
    prediction = "high"
else:
    prediction = "low" if P(low) >= P(medium) else "medium"
```

---

## Final model

**Model:** `HistGradientBoostingClassifier` inside a sklearn `Pipeline`

**Pipeline steps:**
1. `ColumnTransformer` — StandardScaler for numerics, OneHotEncoder for categoricals
2. `HistGradientBoostingClassifier`

**Saved as a bundle** at `models/no_location_baselines/best_no_location_threshold_model.joblib`:

```python
{
    "pipeline":           <sklearn Pipeline>,
    "selected_threshold": 0.2,
    "class_labels":       ["high", "low", "medium"],
    "feature_config":     { ... }
}
```

**Final metrics (group-city test set, threshold=0.2):**

| Metric | Value |
|---|---:|
| Accuracy | 0.8837 |
| Macro F1 | 0.8163 |
| Weighted F1 | 0.8832 |
| High precision | 0.8015 |
| High recall | 0.5769 |
| High F1 | 0.6709 |

The model uses **36 no-location features** — city, latitude, and longitude are excluded from model input. They are used only for row lookup in the backend.

---

## Streamlit app

The hosted demo uses **Streamlit** and lives in:

```text
app.py
```

The Streamlit app is the easiest way to run or deploy the project because it
does **not** require the large CSV/model files to be committed to GitHub. At
startup, it downloads the required artifacts directly from Hugging Face:

| Artifact | Hugging Face source | Used for |
|---|---|---|
| `best_no_location_threshold_model.joblib` | `avnisinghal001/narisafe-risk-awareness-model` | Final trained model bundle |
| `feature_config_no_location.json` | `avnisinghal001/narisafe-risk-awareness-model` | Exact no-location model feature columns |
| `narisafe_ml_features.csv` | `avnisinghal001/narisafe-risk-awareness-dataset` | Lookup data for valid city/day/hour/area/complaint combinations |

The Streamlit app:

- loads model/data/config from Hugging Face with `hf_hub_download`
- caches model loading with `st.cache_resource`
- caches dataset/config loading with `st.cache_data`
- uses dependent dropdowns so users select only valid combinations
- hides private or unsuitable complaint categories from the public demo
- applies the same high-risk threshold rule as the final model
- shows risk-awareness level, model confidence, context breakdown, risk factors,
  mitigating factors and data limitations

### Run Streamlit locally

```bash
source .venv/bin/activate
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

Hosted app URL:

```text
https://narisafe-risk-awareness.streamlit.app/
```

---

## Backend API

Built with **FastAPI**. File: `backend/main.py`.

### How it works at startup

1. Loads `models/no_location_baselines/best_no_location_threshold_model.joblib` — extracts pipeline, threshold, class labels
2. Reads `data/processed/ml_ready_no_location/feature_config_no_location.json` — gets the 36 feature column names
3. Loads `data/processed/ml_ready/narisafe_ml_features.csv` — indexes it by `(city, day_of_week, hour, area_context, complaint_type_clean)` for O(1) lookup

### Prediction flow

```
POST /predict  {city, day_of_week, hour, area_context, complaint_type_clean}

  1. Look up the pre-built feature row from the indexed CSV
     (this gives all 40 feature values including city-level stats and OSM data)

  2. Build X using only the 36 no-location feature columns

  3. pipeline.predict_proba(X) → [P(high), P(low), P(medium)]

  4. Apply threshold:
       if P(high) >= 0.2 → "high"
       else → max(P(low), P(medium))

  5. Build explanation from the full row (key risk factors, mitigating factors,
     data limitations)

  6. Return response
```

### /predict response

```json
{
  "risk_awareness_level": "high",
  "model_confidence": {
    "high":   0.4521,
    "low":    0.0312,
    "medium": 0.5167
  },
  "selected_threshold": 0.2,
  "context": {
    "city_context":       { ... },
    "time_context":       { ... },
    "area_context":       { ... },
    "complaint_context":  { ... },
    "infrastructure":     { ... }
  },
  "key_risk_factors":   ["Late-night hour — highest-risk time window", "..."],
  "mitigating_factors": ["Good police access — nearest station 0.75 km", "..."],
  "data_limitations":   ["Street-light data sparse in OSM", "..."],
  "disclaimer":         "This is a public-data-based risk-awareness prototype..."
}
```

Note: probabilities are called `model_confidence`, not "crime probability" or "chance of crime".

---

## Frontend

Single-page UI at `frontend/index.html`. No build step required.

**What it shows:**
- Dropdown inputs: city, day, hour, area type, complaint type
- Risk awareness level badge (low / medium / high)
- Model confidence breakdown
- Key risk factors and mitigating factors (with icons)
- Data limitations
- Infrastructure context card (lighting, police, transport)
- City-level and complaint-level context
- Disclaimer

**Lighting display:**
- If `lighting_data_available = true` → shows score like `3/5`
- If `lighting_data_available = false` → shows `Unknown / estimated (OSM data unavailable)`

---

## Installation

### Requirements

- Python 3.10 or higher
- pip

### Step 1 — Clone the repo

```bash
git clone https://github.com/<your-username>/NariSafe.git
cd NariSafe
```

### Step 2 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Choose how you want to run it

For the **Streamlit app**, no manual artifact download is needed. `app.py`
downloads the model, feature config and lookup CSV from Hugging Face when the
app starts.

For the **FastAPI backend**, download the required files into the expected
local folders:

```bash
# Install HF hub if not already installed
pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="avnisinghal001/narisafe-risk-awareness-dataset",
    filename="narisafe_ml_features.csv",
    repo_type="dataset",
    local_dir="data/processed/ml_ready/",
)

hf_hub_download(
    repo_id="avnisinghal001/narisafe-risk-awareness-model",
    filename="best_no_location_threshold_model.joblib",
    local_dir="models/no_location_baselines/",
)

hf_hub_download(
    repo_id="avnisinghal001/narisafe-risk-awareness-model",
    filename="feature_config_no_location.json",
    local_dir="data/processed/ml_ready_no_location/",
)
EOF
```

> The no-location CSV (`narisafe_ml_features_no_location.csv`) is only needed if you want to re-train. The backend uses the full `narisafe_ml_features.csv` for lookup.

### Step 5 (optional) — Rebuild the dataset from scratch

Only needed if you want to re-run the full pipeline from the raw PDFs and OSM data.

You need:
- `data/raw/crime_women_citywise_2021_2023.pdf`
- `data/raw/crime_women_headwise_citywise_2023.pdf`
- `data/raw/india-latest.osm.pbf` — download from [Geofabrik India](https://download.geofabrik.de/asia/india.html)
- `data/raw/city_coordinates.csv` — required for OSM extraction; not tracked because `data/raw/` is ignored

Then run scripts in order:

```bash
python scripts/01_check_osm_setup.py
python scripts/02_extract_osm_features.py     # slow — reads the full PBF
python scripts/03_extract_citywise_women_crime.py
python scripts/04_extract_all_headwise_women_crime.py
python scripts/05_clean_headwise_for_model.py
python scripts/06_generate_final_dataset.py
python scripts/07_create_model_training_dataset.py
python scripts/08_audit_training_dataset.py
python scripts/09_prepare_ml_features.py
python scripts/10_create_train_test_splits.py
python scripts/11_train_baseline_models.py
python scripts/12_feature_importance.py
python scripts/14_create_no_location_features.py
python scripts/15_create_no_location_splits.py
python scripts/16_train_no_location_baselines.py
python scripts/17_tune_high_risk_threshold.py
python scripts/18_predict_threshold_risk.py
python scripts/19_create_final_ml_summary.py
```

> Script 02 (OSM extraction) can take 30–90 minutes depending on your machine. The PBF is ~1.6GB and each city requires a spatial query.

---

## Running the app

### Option 1 — Streamlit app

Run from the project root:

```bash
source .venv/bin/activate
streamlit run app.py
```

Streamlit will download the model and lookup dataset from Hugging Face on first
load. Open the URL shown in the terminal, usually:

```text
http://localhost:8501
```

This is the same entry point used for Streamlit hosting.

### Option 2 — FastAPI backend + HTML frontend

Run from the project root (not from inside `backend/`):

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload
```

Then open `http://localhost:8000` in your browser.

The frontend is served at `/` and the API docs are at `http://localhost:8000/docs`.

---

## API reference

### `GET /health`

Returns model and dataset status.

```json
{
  "status": "ok",
  "model_type": "HistGradientBoostingClassifier",
  "model_classes": ["high", "low", "medium"],
  "selected_threshold": 0.2,
  "dataset_rows": 489804,
  "feature_count": 36
}
```

### `GET /metadata`

Returns all valid dropdown values for the UI.

```json
{
  "cities": ["Agra", "Amritsar", ...],
  "days": ["Monday", "Tuesday", ...],
  "hours": [6, 9, 12, 15, 18, 21, 23],
  "area_contexts": ["commercial", "educational", ...],
  "complaint_types": [
    {"value": "acid_attack", "label": "Acid Attack"},
    ...
  ]
}
```

### `POST /predict`

**Request body:**

```json
{
  "city": "Gwalior",
  "day_of_week": "Saturday",
  "hour": 23,
  "area_context": "transit",
  "complaint_type_clean": "assault_women_18_and_above"
}
```

**Response:** see [/predict response](#predict-response) above.

---

## What is tracked in git

```
Tracked (in git):
  app.py
  backend/main.py
  frontend/index.html
  scripts/01 – 19
  requirements.txt
  .gitignore
  README.md
  data/processed/ml_ready/feature_config.json
  data/processed/ml_ready_no_location/feature_config_no_location.json
  reports/ml/feature_importance.csv
  reports/ml_no_location/baseline_model_report.md
  reports/ml_no_location/high_threshold_tuning.csv
  reports/final/final_ml_summary.md
  reports/final/final_model_manifest.json

NOT tracked (too large or generated):
  .venv/
  data/raw/                    (PDFs, OSM PBF — large, manual download)
  data/intermediate/           (generated by scripts 02–05)
  data/processed/*.csv         (generated — download from HF instead)
  data/processed/ml_ready/*.csv
  data/processed/ml_ready_no_location/*.csv
  data/processed/ml_ready/splits/
  data/processed/ml_ready_no_location/splits/
  data/processed/audit/
  models/baselines/            (joblib files — download from HF)
  models/no_location_baselines/*.joblib
  hf_upload/                   (upload staging folder)
  __pycache__/
```

---

## Limitations

- **Proxy labels, not ground truth.** `risk_level` was computed from a rule-based scoring formula, not from real incident-level police data. The model learns patterns from these proxy labels.
- **City-level, not hyperlocal.** All infrastructure and crime features are aggregated at the city level. Two locations in the same city get the same OSM features.
- **OSM data quality varies.** Street light data in particular is sparse for many Indian cities. If `lighting_data_available = false`, lighting scores are estimated.
- **NCRB data is annual, not real-time.** Crime statistics are from 2021–2023. The model does not have access to recent or live data.
- **Model confidence ≠ crime probability.** The numbers returned as `model_confidence` are classifier probabilities on proxy labels. They do not represent the real-world probability of any crime occurring.
- **Not for emergency decisions.** Do not use this tool for policing, emergency response, official safety advisories, or any decision affecting individual risk.

---

## Tech stack

| Layer | Technology |
|---|---|
| Data extraction | `pdfplumber`, `pyrosm`, `geopandas`, `shapely` |
| Data processing | `pandas`, `numpy` |
| ML | `scikit-learn` (HistGradientBoosting, RandomForest, LogisticRegression) |
| Model serialization | `joblib` |
| Backend | `FastAPI`, `uvicorn` |
| Hosted app | `Streamlit` |
| Frontend | Streamlit app (`app.py`) and vanilla HTML / CSS / JS fallback (`frontend/index.html`) |
| Model hosting | Hugging Face Hub |
| Dataset hosting | Hugging Face Hub |

---

## License

MIT

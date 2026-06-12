# NariSafe ML Feature Preparation Report

## Input dataset

`data/processed/narisafe_model_training_dataset.csv`

## Output dataset

`data/processed/ml_ready/narisafe_ml_features.csv`

## Shape

Input shape: `(489804, 40)`  
Output shape: `(489804, 40)`

## Target column

`risk_level`

## Categorical features

- `city`
- `day_of_week`
- `time_bucket`
- `area_context`
- `complaint_type_clean`
- `crowd_density`

## Numeric features

- `latitude`
- `longitude`
- `is_weekend`
- `hour`
- `complaint_type_count`
- `complaint_type_share`
- `women_crime_2021`
- `women_crime_2022`
- `women_crime_2023`
- `population_lakhs`
- `women_crime_rate_2023`
- `chargesheeting_rate_2023`
- `women_crime_growth_21_23`
- `police_station_count_5km`
- `nearest_police_station_km`
- `public_transport_count_3km`
- `bus_stop_count_3km`
- `bus_station_count_3km`
- `railway_station_count_3km`
- `street_light_count_2km`
- `road_length_km_5km`
- `road_density_km_per_sqkm_5km`
- `commercial_landuse_count_5km`
- `residential_landuse_count_5km`
- `industrial_landuse_count_5km`
- `retail_landuse_count_5km`
- `education_poi_count_5km`
- `lighting_data_available`
- `lighting_score`
- `police_access_score`
- `transport_access_score`
- `urban_density_score`
- `complaint_severity`

## Ignored columns

None

## Important ML notes

- `risk_level` is a rule-based proxy label.
- `risk_score` must not be used as an input feature.
- Use group split by city for better evaluation.
- Accuracy alone is not enough because the `high` class is small.
- Use macro F1, weighted F1, confusion matrix, and per-class recall.

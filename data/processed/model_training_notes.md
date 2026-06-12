# NariSafe — Model Training Notes

## Dataset Overview

| Property | Value |
|---|---|
| File | `data/processed/narisafe_city_time_risk_dataset.csv` |
| Rows | 489,804 |
| Cities | 34 |
| Days | 7 (Monday – Sunday) |
| Hours | 7 (6, 9, 12, 15, 18, 21, 23) |
| Area contexts | 7 (residential, commercial, market, industrial, transit, educational, mixed) |
| Complaint types | 42 |
| Label source | rule_based_public_data_proxy |

## Data Sources

| Source | File | Description |
|---|---|---|
| NCRB 2021-2023 | `crime_women_citywise_2021_2023.pdf` | City-level women crime counts + rates |
| NCRB 2023 | `crime_women_headwise_citywise_2023.pdf` | Crime-head breakdown (42 types) |
| OpenStreetMap | `india-latest.osm.pbf` | Infrastructure features (police, transit, lighting, roads) |

## Target Variable

`risk_level` — ordinal, 3 classes: **low / medium / high**

| Class | Condition | Count |
|---|---|---|
| low | risk_score ≤ 6 | 177,702 |
| medium | risk_score 7–12 | 287,812 |
| high | risk_score ≥ 13 | 24,290 |

> **Warning:** `risk_score` encodes the same information as `risk_level`.
> Do **not** use `risk_score` as an input feature when `risk_level` is the target.
> Drop `risk_score` (and `label_source`) from X before training.

## Feature Groups

### Identifiers (drop before training)
`city`, `latitude`, `longitude`, `day_of_week`, `label_source`

### Temporal features
| Feature | Type | Notes |
|---|---|---|
| `is_weekend` | int (0/1) | 1 = Saturday/Sunday |
| `hour` | int | One of: [6, 9, 12, 15, 18, 21, 23] |
| `time_bucket` | str | morning / afternoon / evening / late_night |

### Complaint / crime-type features
| Feature | Type | Notes |
|---|---|---|
| `complaint_type_clean` | str (cat) | 42 atomic types, no totals |
| `complaint_type_count` | int | Raw NCRB 2023 incidence count for city |
| `complaint_type_share` | float | complaint_type_count / women_crime_2023 |
| `complaint_severity` | int (1–3) | Rule-based severity; 3=most severe |

### City-level crime statistics (raw)
`women_crime_2021`, `women_crime_2022`, `women_crime_2023`,
`population_lakhs`, `women_crime_rate_2023`, `chargesheeting_rate_2023`,
`women_crime_growth_21_23`

### OSM infrastructure (raw)
`police_station_count_5km`, `nearest_police_station_km`,
`public_transport_count_3km`, `bus_stop_count_3km`, `bus_station_count_3km`,
`railway_station_count_3km`, `street_light_count_2km`,
`road_length_km_5km`, `road_density_km_per_sqkm_5km`,
`commercial_landuse_count_5km`, `residential_landuse_count_5km`,
`industrial_landuse_count_5km`, `retail_landuse_count_5km`,
`education_poi_count_5km`

> **Street light note:** `street_light_count_2km` is 0 for most cities — this
> reflects missing OSM data, not actual darkness. The `lighting_score` treats
> 0 as moderate (score=3), not as dark.

### Engineered scores (helper / explainability — keep raw OSM features too)
| Feature | Range | Derived from |
|---|---|---|
| `lighting_score` | 1–5 | street_light_count_2km + area_context + time_bucket |
| `crowd_density` | low/medium/high | area_context + hour |
| `police_access_score` | 0–3 | nearest_police_station_km |
| `transport_access_score` | 0–3 | public_transport_count_3km |
| `urban_density_score` | 0–3 | road_density_km_per_sqkm_5km |

### Risk features (target)
| Feature | Notes |
|---|---|
| `risk_score` | Raw additive score (int); drop from X if risk_level is target |
| `risk_level` | Final label: low / medium / high |

## Risk Score Formula

```
risk_score =
  # city-level crime signal
  + [3 if crime_rate≥200, 2 if ≥100, 1 if ≥50]
  + [1 if crime_growth > 10%]
  # complaint signal
  + [3 if share≥0.20, 2 if ≥0.05, 1 if >0]
  + complaint_severity          # 1-3
  # time signal
  + [2 if late_night, 1 if evening]
  # area+time interaction
  + [2 if (industrial/transit) & (evening/late_night)]
  + [1 if (market/commercial) & evening]
  + [1 if educational & hour 9-18]
  # infrastructure signal
  + [2 if lighting_score≤2, 1 if =3]
  + [2 if crowd=low, 1 if crowd=medium]
  + [2 if police_access=0, 1 if =1]
  + [2 if transport_access=0, 1 if =1]
```

## Known Limitations

1. **Synthetic spatio-temporal rows**: hours, days, and area contexts are
   enumerated, not from actual incident timestamps.  Risk scores are
   rule-based proxies, not ground-truth labels.

2. **City-level aggregation**: all rows for the same city share the same
   infrastructure and crime-rate features — within-city spatial variation
   is not captured.

3. **OSM data quality**: street-light counts are 0 for most cities (missing
   OSM coverage), and police-station counts may under-represent actual
   deployment.

4. **Class imbalance**: the risk label distribution is dominated by
   "medium" rows (287,812 / 489,804).
   Use stratified splits and appropriate evaluation metrics (macro-F1,
   balanced accuracy) when training.

5. **Temporal leakage**: crime statistics are from 2021-2023; the synthetic
   time dimension (hour, day) does not map to those actual years.

## Recommended Training Splits

- **Stratified train/test split** on `risk_level` (80/20)
- Hold out 3–5 cities entirely for geo-generalisation testing
- Use cross-validation folds stratified on both city and risk_level

## Suggested Baseline Models

1. Random Forest (handles mixed types well, interpretable feature importance)
2. Gradient Boosting (XGBoost / LightGBM) — strong on tabular data
3. Logistic Regression on one-hot encoded categoricals (interpretable baseline)

Encode categoricals: `complaint_type_clean`, `area_context`, `time_bucket`,
`crowd_density` as one-hot or ordinal depending on model type.

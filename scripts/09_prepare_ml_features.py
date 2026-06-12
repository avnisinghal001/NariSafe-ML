import json
from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/narisafe_model_training_dataset.csv")
OUTPUT_DIR = Path("data/processed/ml_ready")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DATASET = OUTPUT_DIR / "narisafe_ml_features.csv"
OUTPUT_CONFIG = OUTPUT_DIR / "feature_config.json"
OUTPUT_REPORT = OUTPUT_DIR / "feature_preparation_report.md"


TARGET_COLUMN = "risk_level"

CATEGORICAL_FEATURES = [
    "city",
    "day_of_week",
    "time_bucket",
    "area_context",
    "complaint_type_clean",
    "crowd_density",
]

NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "is_weekend",
    "hour",
    "complaint_type_count",
    "complaint_type_share",
    "women_crime_2021",
    "women_crime_2022",
    "women_crime_2023",
    "population_lakhs",
    "women_crime_rate_2023",
    "chargesheeting_rate_2023",
    "women_crime_growth_21_23",
    "police_station_count_5km",
    "nearest_police_station_km",
    "public_transport_count_3km",
    "bus_stop_count_3km",
    "bus_station_count_3km",
    "railway_station_count_3km",
    "street_light_count_2km",
    "road_length_km_5km",
    "road_density_km_per_sqkm_5km",
    "commercial_landuse_count_5km",
    "residential_landuse_count_5km",
    "industrial_landuse_count_5km",
    "retail_landuse_count_5km",
    "education_poi_count_5km",
    "lighting_data_available",
    "lighting_score",
    "police_access_score",
    "transport_access_score",
    "urban_density_score",
    "complaint_severity",
]


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    print("\n================ ML FEATURE PREPARATION ================\n")

    print("Input shape:", df.shape)

    required_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET_COLUMN]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    extra_cols = [col for col in df.columns if col not in required_cols]

    ml_df = df[required_cols].copy()

    print("\nTarget column:")
    print(TARGET_COLUMN)

    print("\nCategorical features:")
    for col in CATEGORICAL_FEATURES:
        print(f"- {col}: {ml_df[col].nunique()} unique values")

    print("\nNumeric features:")
    for col in NUMERIC_FEATURES:
        print(f"- {col}")

    print("\nExtra columns ignored:")
    if extra_cols:
        for col in extra_cols:
            print(f"- {col}")
    else:
        print("None")

    print("\nMissing values in final ML dataset:")
    missing = ml_df.isnull().sum()
    print(missing[missing > 0])

    print("\nTarget distribution:")
    print(ml_df[TARGET_COLUMN].value_counts())
    print("\nTarget percentage:")
    print((ml_df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2))

    print("\nFinal ML dataset shape:")
    print(ml_df.shape)

    ml_df.to_csv(OUTPUT_DATASET, index=False)

    config = {
        "target_column": TARGET_COLUMN,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "ignored_columns": extra_cols,
        "notes": [
            "risk_level is the target column.",
            "risk_score must not be used because it causes target leakage.",
            "label_source must not be used because it is constant.",
            "complaint_type_original must not be used because complaint_type_clean already exists.",
            "To use group split by city for realistic evaluation.",
            "Not rely on accuracy alone because high-risk class is underrepresented."
        ]
    }

    with open(OUTPUT_CONFIG, "w") as f:
        json.dump(config, f, indent=2)

    report = f"""# NariSafe ML Feature Preparation Report

## Input dataset

`{INPUT_PATH}`

## Output dataset

`{OUTPUT_DATASET}`

## Shape

Input shape: `{df.shape}`  
Output shape: `{ml_df.shape}`

## Target column

`{TARGET_COLUMN}`

## Categorical features

{chr(10).join([f"- `{col}`" for col in CATEGORICAL_FEATURES])}

## Numeric features

{chr(10).join([f"- `{col}`" for col in NUMERIC_FEATURES])}

## Ignored columns

{chr(10).join([f"- `{col}`" for col in extra_cols]) if extra_cols else "None"}

## Important ML notes

- `risk_level` is a rule-based proxy label.
- `risk_score` must not be used as an input feature.
- Use group split by city for better evaluation.
- Accuracy alone is not enough because the `high` class is small.
- To use macro F1, weighted F1, confusion matrix and per-class recall.
"""

    with open(OUTPUT_REPORT, "w") as f:
        f.write(report)

    print("\nSaved:")
    print(OUTPUT_DATASET)
    print(OUTPUT_CONFIG)
    print(OUTPUT_REPORT)

    print("\n================ FEATURE PREPARATION COMPLETE ================\n")


if __name__ == "__main__":
    main()
"""
Create a cleaner ML training version of the master dataset.

Input  : data/processed/narisafe_city_time_risk_dataset.csv
Output : data/processed/narisafe_model_training_dataset.csv

The master dataset is never modified.
"""

import sys
import pandas as pd

INPUT_PATH  = "data/processed/narisafe_city_time_risk_dataset.csv"
OUTPUT_PATH = "data/processed/narisafe_model_training_dataset.csv"

DROP_COLUMNS = [
    "complaint_type_original",  # duplicate of complaint_type_clean (human-readable text)
    "risk_score",               # target was derived from this — must not be an input feature
    "label_source",             # constant column; no learning value
]

KEEP_COLUMNS = [
    # --- identifiers / categoricals ---
    "city",
    "day_of_week",
    "time_bucket",
    "area_context",
    "complaint_type_clean",
    "crowd_density",
    # --- numeric / raw ---
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
    "lighting_data_available",      # added below
    "lighting_score",
    "police_access_score",
    "transport_access_score",
    "urban_density_score",
    "complaint_severity",
    # --- target ---
    "risk_level",
]


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    print(f"Original shape : {df.shape}")

    # ── Add lighting_data_available if absent ─────────────────────────────────
    if "lighting_data_available" not in df.columns:
        df["lighting_data_available"] = (df["street_light_count_2km"] > 0).astype(int)
        print("Added column   : lighting_data_available")
    else:
        print("Column already present: lighting_data_available")

    # ── Verify DROP columns exist before dropping ─────────────────────────────
    missing_drops = [c for c in DROP_COLUMNS if c not in df.columns]
    if missing_drops:
        print(f"[WARN] Columns to drop not found (already absent): {missing_drops}")
    actually_dropped = [c for c in DROP_COLUMNS if c in df.columns]

    df = df.drop(columns=actually_dropped)
    print(f"Columns removed: {actually_dropped}")

    # ── Select and order final columns ────────────────────────────────────────
    missing_keep = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing_keep:
        print(f"[ERROR] Expected columns not found in dataset: {missing_keep}")
        sys.exit(1)

    df = df[KEEP_COLUMNS]

    # ── Validation ────────────────────────────────────────────────────────────
    errors = []

    if "risk_score" in df.columns:
        errors.append("risk_score must not be present in the training dataset")
    if "label_source" in df.columns:
        errors.append("label_source must not be present in the training dataset")
    if "complaint_type_original" in df.columns:
        errors.append("complaint_type_original must not be present in the training dataset")
    if "risk_level" not in df.columns:
        errors.append("risk_level (target) is missing")

    missing_vals = df.isna().sum()
    if missing_vals.any():
        errors.append(f"Missing values detected:\n{missing_vals[missing_vals > 0]}")

    if errors:
        for e in errors:
            print(f"\n[VALIDATION ERROR] {e}")
        sys.exit(1)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print(f"\nCleaned shape  : {df.shape}")

    print(f"\nFinal columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  {col}")

    print("\nMissing values : none")

    print("\nrisk_level distribution:")
    print(df["risk_level"].value_counts().sort_index().to_string())

    print(f"\nUnique cities                 : {df['city'].nunique()}")
    print(f"Unique complaint_type_clean   : {df['complaint_type_clean'].nunique()}")

    print("\nlighting_data_available distribution:")
    print(df["lighting_data_available"].value_counts().sort_index().to_string())

    print("\nFirst 10 rows:")
    pd.set_option("display.max_columns", 8)
    pd.set_option("display.width", 160)
    print(df.head(10).to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df):,} rows × {len(df.columns)} columns → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

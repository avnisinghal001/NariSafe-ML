"""
Generate the final ML-ready dataset for NariSafe.

Inputs:
  data/intermediate/osm_feature_summary.csv
  data/intermediate/women_crime_citywise.csv
  data/intermediate/women_crime_headwise_model_ready.csv

Output:
  data/processed/narisafe_city_time_risk_dataset.csv
  data/processed/model_training_notes.md
"""

import sys
import numpy as np
import pandas as pd

OSM_PATH     = "data/intermediate/osm_feature_summary.csv"
CITYWISE_PATH = "data/intermediate/women_crime_citywise.csv"
HEADWISE_PATH = "data/intermediate/women_crime_headwise_model_ready.csv"
OUTPUT_CSV   = "data/processed/narisafe_city_time_risk_dataset.csv"
OUTPUT_NOTES = "data/processed/model_training_notes.md"

# ── Dimension constants ────────────────────────────────────────────────────────
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
HOURS = [6, 9, 12, 15, 18, 21, 23]
AREA_CONTEXTS = [
    "residential", "commercial", "market",
    "industrial", "transit", "educational", "mixed",
]

COMPLAINT_SEVERITY_MAP = {
    # severity 3 — most severe
    "murder_with_rape_gang_rape_ipc":   3,
    "acid_attack":                      3,
    "attempt_to_acid_attack":           3,
    "human_trafficking":                3,
    "selling_of_minor_girls":           3,
    "buying_of_minor_girls":            3,
    "rape_women_18_and_above":          3,
    "rape_girls_below_18":              3,
    "attempt_rape_women_18_and_above":  3,
    "attempt_rape_girls_below_18":      3,
    "pocso_child_rape":                 3,
    # severity 2
    "dowry_deaths":                     2,
    "kidnapping_and_abduction_sec_363": 2,
    "kidnapping_and_abduction_to_murder": 2,
    "kidnapping_for_ransom":            2,
    "kna_for_marriage_women_above_18":  2,
    "kna_for_marriage_girls_below_18":  2,
    "importation_of_girls_foreign_country": 2,
    "procuration_of_minor_girls":       2,
    "kidnapping_abduction_others":      2,
    "assault_women_18_and_above":       2,
    "assault_girls_below_18":           2,
    "pocso_sexual_assault":             2,
    "pocso_sexual_harassment":          2,
    "abetment_to_suicide_of_women":     2,
    "miscarriage":                      2,
    # severity 1
    "cruelty_by_husband_or_relatives":          1,
    "dowry_prohibition_act":                    1,
    "insult_modesty_women_18_and_above":        1,
    "insult_modesty_girls_below_18":            1,
    "itp_procuring_inducing_children":          1,
    "itp_detaining_in_premises":                1,
    "itp_prostitution_near_public_places":      1,
    "itp_seducing_soliciting":                  1,
    "itp_other_sections":                       1,
    "domestic_violence_act":                    1,
    "cyber_publishing_explicit_material":       1,
    "cyber_other_women_centric":                1,
    "pocso_child_pornography":                  1,
    "pocso_other_offences":                     1,
    "pocso_unnatural_offences":                 1,
    "indecent_representation_of_women_act":     1,
}

FINAL_COLUMNS = [
    "city", "latitude", "longitude",
    "day_of_week", "is_weekend", "hour", "time_bucket", "area_context",
    "complaint_type_original", "complaint_type_clean",
    "complaint_type_count", "complaint_type_share",
    "women_crime_2021", "women_crime_2022", "women_crime_2023",
    "population_lakhs", "women_crime_rate_2023",
    "chargesheeting_rate_2023", "women_crime_growth_21_23",
    "police_station_count_5km", "nearest_police_station_km",
    "public_transport_count_3km", "bus_stop_count_3km",
    "bus_station_count_3km", "railway_station_count_3km",
    "street_light_count_2km", "road_length_km_5km",
    "road_density_km_per_sqkm_5km",
    "commercial_landuse_count_5km", "residential_landuse_count_5km",
    "industrial_landuse_count_5km", "retail_landuse_count_5km",
    "education_poi_count_5km",
    "lighting_score", "crowd_density",
    "police_access_score", "transport_access_score", "urban_density_score",
    "complaint_severity",
    "risk_score", "risk_level", "label_source",
]


# ── Feature engineering functions (all vectorised) ────────────────────────────

def add_is_weekend(df: pd.DataFrame) -> pd.DataFrame:
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    return df


def add_time_bucket(df: pd.DataFrame) -> pd.DataFrame:
    h = df["hour"]
    df["time_bucket"] = np.select(
        [h.between(6, 11), h.between(12, 16), h.between(17, 21)],
        ["morning",        "afternoon",       "evening"],
        default="late_night",  # 22-5; only hour=23 in our set
    )
    return df


def add_lighting_score(df: pd.DataFrame) -> pd.DataFrame:
    sl = df["street_light_count_2km"]
    # Base score; 0 → 3 (unknown/moderate — OSM street-light data is sparse,
    # a zero count means "no OSM data" not "actually dark")
    base = np.select(
        [sl >= 50, sl >= 20, sl >= 5, sl >= 1, sl == 0],
        [5,        4,        3,       2,        3],
        default=3,
    )
    score = base.astype(float)

    # -1 for high-risk area+time combinations
    mask_minus = (
        df["area_context"].isin(["industrial", "transit"])
        & df["time_bucket"].isin(["evening", "late_night"])
    )
    score = np.where(mask_minus, score - 1, score)

    # +1 for well-lit area during active hours
    mask_plus = (
        df["area_context"].isin(["commercial", "market", "educational"])
        & df["hour"].between(9, 18)
    )
    score = np.where(mask_plus, score + 1, score)

    df["lighting_score"] = np.clip(score, 1, 5).astype(int)
    return df


def add_crowd_density(df: pd.DataFrame) -> pd.DataFrame:
    area = df["area_context"]
    h    = df["hour"]
    tb   = df["time_bucket"]

    crowd = np.full(len(df), "low", dtype=object)

    # Base rules (applied in ascending priority order; later assignments win)
    crowd = np.where(area.eq("residential") & h.between(6, 21),
                     "medium", crowd)
    crowd = np.where(area.eq("industrial") & h.between(6, 18),
                     "medium", crowd)
    crowd = np.where(area.eq("mixed") & tb.isin(["morning", "afternoon", "evening"]),
                     "medium", crowd)
    crowd = np.where(area.isin(["market", "commercial", "transit"]) & h.between(9, 21),
                     "high", crowd)
    crowd = np.where(area.eq("educational") & h.between(9, 18),
                     "high", crowd)

    # Industrial late = low (overrides the medium set above)
    crowd = np.where(area.eq("industrial") & h.isin([21, 23]),
                     "low", crowd)

    # hour=23: everything → low, except transit → medium
    crowd = np.where(h.eq(23), "low", crowd)
    crowd = np.where(h.eq(23) & area.eq("transit"), "medium", crowd)

    df["crowd_density"] = crowd
    return df


def add_police_access_score(df: pd.DataFrame) -> pd.DataFrame:
    d = df["nearest_police_station_km"]
    df["police_access_score"] = np.select(
        [d <= 1, d <= 3, d <= 5],
        [3,      2,      1],
        default=0,
    ).astype(int)
    return df


def add_transport_access_score(df: pd.DataFrame) -> pd.DataFrame:
    t = df["public_transport_count_3km"]
    df["transport_access_score"] = np.select(
        [t >= 30, t >= 10, t >= 1],
        [3,       2,       1],
        default=0,
    ).astype(int)
    return df


def add_urban_density_score(df: pd.DataFrame) -> pd.DataFrame:
    r = df["road_density_km_per_sqkm_5km"]
    df["urban_density_score"] = np.select(
        [r >= 20, r >= 12, r >= 6],
        [3,       2,       1],
        default=0,
    ).astype(int)
    return df


def add_complaint_severity(df: pd.DataFrame) -> pd.DataFrame:
    unmapped = set(df["complaint_type_clean"].unique()) - set(COMPLAINT_SEVERITY_MAP)
    if unmapped:
        for ct in sorted(unmapped):
            print(f"  [WARN] complaint_type_clean not in severity map — defaulting to 1: {ct}")
    df["complaint_severity"] = (
        df["complaint_type_clean"]
        .map(COMPLAINT_SEVERITY_MAP)
        .fillna(1)
        .astype(int)
    )
    return df


def add_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    risk = pd.Series(0, index=df.index, dtype=int)

    # Crime rate component
    risk += np.select(
        [df["women_crime_rate_2023"] >= 200,
         df["women_crime_rate_2023"] >= 100,
         df["women_crime_rate_2023"] >= 50],
        [3, 2, 1], default=0,
    )

    # Crime growth
    risk += (df["women_crime_growth_21_23"] > 0.10).astype(int)

    # Complaint share
    risk += np.select(
        [df["complaint_type_share"] >= 0.20,
         df["complaint_type_share"] >= 0.05,
         df["complaint_type_share"] > 0],
        [3, 2, 1], default=0,
    )

    # Complaint severity (already computed)
    risk += df["complaint_severity"]

    # Time of day
    risk += np.select(
        [df["time_bucket"] == "late_night",
         df["time_bucket"] == "evening"],
        [2, 1], default=0,
    )

    # Area context (mutually exclusive by area_context value)
    risk += np.select(
        [
            df["area_context"].isin(["industrial", "transit"])
            & df["time_bucket"].isin(["evening", "late_night"]),
            df["area_context"].isin(["market", "commercial"])
            & (df["time_bucket"] == "evening"),
            df["area_context"].eq("educational") & df["hour"].between(9, 18),
        ],
        [2, 1, 1], default=0,
    )

    # Lighting
    risk += np.select(
        [df["lighting_score"] <= 2, df["lighting_score"] == 3],
        [2, 1], default=0,
    )

    # Crowd
    risk += np.select(
        [df["crowd_density"] == "low", df["crowd_density"] == "medium"],
        [2, 1], default=0,
    )

    # Police access
    risk += np.select(
        [df["police_access_score"] == 0, df["police_access_score"] == 1],
        [2, 1], default=0,
    )

    # Transport access
    risk += np.select(
        [df["transport_access_score"] == 0, df["transport_access_score"] == 1],
        [2, 1], default=0,
    )

    df["risk_score"] = risk

    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-1, 6, 12, 9999],
        labels=["low", "medium", "high"],
    ).astype(str)

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load inputs ───────────────────────────────────────────────────────────
    osm = pd.read_csv(OSM_PATH)
    cw  = pd.read_csv(CITYWISE_PATH)
    hw  = pd.read_csv(HEADWISE_PATH)

    print("Input shapes:")
    print(f"  OSM            : {osm.shape}")
    print(f"  citywise crime : {cw.shape}")
    print(f"  headwise crime : {hw.shape}")
    print(f"\nUnique cities — OSM: {osm.city.nunique()}, "
          f"citywise: {cw.city.nunique()}, headwise: {hw.city.nunique()}")
    print(f"Unique complaint types (headwise): {hw.complaint_type_clean.nunique()}")

    cities      = sorted(osm["city"].unique())
    ctypes      = sorted(hw["complaint_type_clean"].unique())
    expected    = len(cities) * len(DAYS) * len(HOURS) * len(AREA_CONTEXTS) * len(ctypes)
    print(f"\nExpected rows: {len(cities)} cities × {len(DAYS)} days × "
          f"{len(HOURS)} hours × {len(AREA_CONTEXTS)} areas × "
          f"{len(ctypes)} complaint types = {expected:,}")

    # ── Build skeleton via MultiIndex cross-product ───────────────────────────
    idx = pd.MultiIndex.from_product(
        [cities, DAYS, HOURS, AREA_CONTEXTS, ctypes],
        names=["city", "day_of_week", "hour", "area_context", "complaint_type_clean"],
    )
    df = idx.to_frame(index=False)

    # ── Merge complaint-type data ─────────────────────────────────────────────
    hw_slim = hw[["city", "complaint_type_clean",
                  "complaint_type_original", "complaint_type_count", "complaint_type_share"]]
    df = df.merge(hw_slim, on=["city", "complaint_type_clean"], how="left")

    # ── Merge city-level crime stats ───────────────────────────────────────────
    df = df.merge(cw, on="city", how="left")

    # ── Merge OSM infrastructure ──────────────────────────────────────────────
    osm_cols = [c for c in osm.columns if c != "osm_extraction_notes"]
    df = df.merge(osm[osm_cols], on="city", how="left")

    # ── Feature engineering ───────────────────────────────────────────────────
    df = add_is_weekend(df)
    df = add_time_bucket(df)
    df = add_lighting_score(df)
    df = add_crowd_density(df)
    df = add_police_access_score(df)
    df = add_transport_access_score(df)
    df = add_urban_density_score(df)
    df = add_complaint_severity(df)
    df = add_risk_score(df)

    df["label_source"] = "rule_based_public_data_proxy"

    # ── Reorder columns ───────────────────────────────────────────────────────
    df = df[FINAL_COLUMNS]

    # ── Validation ────────────────────────────────────────────────────────────
    errors = []

    actual = len(df)
    if actual != expected:
        errors.append(f"Row count mismatch: expected {expected:,}, got {actual:,}")

    missing = df.isna().sum()
    missing_nonzero = missing[missing > 0]

    for col in ("complaint_type_count", "complaint_type_share",
                "risk_score", "complaint_severity"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' is not numeric")

    if errors:
        for e in errors:
            print(f"\n[VALIDATION ERROR] {e}")
        sys.exit(1)

    print("\n[ALL VALIDATIONS PASSED]")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print(f"\nActual final row count  : {actual:,}")

    print("\nMissing values:")
    if missing_nonzero.empty:
        print("  none")
    else:
        print(missing_nonzero.to_string())

    print("\nrisk_level distribution:")
    print(df["risk_level"].value_counts().sort_index().to_string())

    print("\ncity distribution (row counts):")
    print(df["city"].value_counts().sort_index().to_string())

    print("\ncomplaint_type_clean distribution (row counts across all city/time combos):")
    print(df["complaint_type_clean"].value_counts().sort_index().to_string())

    print("\ncomplaint_severity distribution:")
    print(df["complaint_severity"].value_counts().sort_index().to_string())

    print("\nlabel_source distribution:")
    print(df["label_source"].value_counts().to_string())

    print("\nFirst 20 rows:")
    pd.set_option("display.max_colwidth", 45)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 160)
    print(df.head(20).to_string(index=False))

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {actual:,} rows → {OUTPUT_CSV}")

    # ── Write training notes ──────────────────────────────────────────────────
    write_training_notes(df, cities, ctypes, actual)
    print(f"Saved training notes → {OUTPUT_NOTES}")


def write_training_notes(df: pd.DataFrame, cities: list, ctypes: list, n_rows: int) -> None:
    risk_dist = df["risk_level"].value_counts().sort_index()

    notes = f"""# NariSafe — Model Training Notes

## Dataset Overview

| Property | Value |
|---|---|
| File | `data/processed/narisafe_city_time_risk_dataset.csv` |
| Rows | {n_rows:,} |
| Cities | {len(cities)} |
| Days | 7 (Monday – Sunday) |
| Hours | 7 ({', '.join(str(h) for h in HOURS)}) |
| Area contexts | {len(AREA_CONTEXTS)} ({', '.join(AREA_CONTEXTS)}) |
| Complaint types | {len(ctypes)} |
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
| low | risk_score ≤ 6 | {risk_dist.get('low', 0):,} |
| medium | risk_score 7–12 | {risk_dist.get('medium', 0):,} |
| high | risk_score ≥ 13 | {risk_dist.get('high', 0):,} |

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
| `hour` | int | One of: {HOURS} |
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
   "{risk_dist.idxmax()}" rows ({risk_dist.max():,} / {n_rows:,}).
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
"""
    with open(OUTPUT_NOTES, "w") as f:
        f.write(notes)


if __name__ == "__main__":
    main()

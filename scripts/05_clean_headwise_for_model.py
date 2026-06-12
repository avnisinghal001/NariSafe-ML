"""
Remove total/subtotal rows that duplicate child categories from the head-wise
crime data, producing a model-ready version with only atomic complaint types.

Input  : data/intermediate/women_crime_headwise_all_types.csv
Output : data/intermediate/women_crime_headwise_model_ready.csv
"""

import sys
import pandas as pd

INPUT_PATH = "data/intermediate/women_crime_headwise_all_types.csv"
OUTPUT_PATH = "data/intermediate/women_crime_headwise_model_ready.csv"

# Rows to drop: totals/subtotals that are arithmetic sums of child categories
# already present in the file (keeping them would double-count in ML features).
DROP_TYPES = {
    "total_ipc_crimes_against_women",
    "total_sll_crimes_against_women",
    "total_crime_against_women",
    "kidnapping_and_abduction_of_women",
    "kna_to_compel_for_marriage_total",
    "rape_total",
    "attempt_to_commit_rape_total",
    "assault_on_women_with_intent_to_outrage_modesty",
    "insult_to_modesty_of_women",
    "immoral_traffic_prevention_act_total",
    "cyber_crimes_against_women_total",
    "pocso_act_total",
}


def main() -> None:
    df_raw = pd.read_csv(INPUT_PATH)
    print(f"Original row count : {len(df_raw)}")

    df = df_raw[~df_raw["complaint_type_clean"].isin(DROP_TYPES)].copy()
    df = df.reset_index(drop=True)
    print(f"Cleaned row count  : {len(df)}")
    print(f"Rows removed       : {len(df_raw) - len(df)}")

    # ── Validation ────────────────────────────────────────────────────────────
    errors = []

    # 3. Unique cities
    n_cities = df["city"].nunique()
    print(f"\nUnique cities      : {n_cities}")
    if n_cities != 34:
        errors.append(f"Expected 34 cities, got {n_cities}")

    # 4. Unique complaint types
    n_types = df["complaint_type_clean"].nunique()
    print(f"Unique complaint types: {n_types}")

    # 5. List of final complaint_type_clean values
    print("\nFinal complaint_type_clean values:")
    for ct in sorted(df["complaint_type_clean"].unique()):
        print(f"  - {ct}")

    # 6. Every city has the same set of complaint types
    all_types = set(df["complaint_type_clean"].unique())
    for city, grp in df.groupby("city"):
        city_types = set(grp["complaint_type_clean"])
        missing = all_types - city_types
        extra = city_types - all_types
        if missing or extra:
            errors.append(f"{city}: missing={missing}, extra={extra}")

    # 7. No duplicate city + complaint_type_clean pairs
    dupes = df[df.duplicated(["city", "complaint_type_clean"])]
    if not dupes.empty:
        errors.append(
            f"Duplicate city+type pairs: "
            f"{dupes[['city', 'complaint_type_clean']].values.tolist()}"
        )

    # 8. No missing values
    missing_vals = df.isna().sum()
    if missing_vals.any():
        errors.append(f"Missing values:\n{missing_vals[missing_vals > 0]}")

    # 9. complaint_type_count is numeric
    if not pd.api.types.is_numeric_dtype(df["complaint_type_count"]):
        errors.append("complaint_type_count is not numeric")

    # 10. complaint_type_share is numeric
    if not pd.api.types.is_numeric_dtype(df["complaint_type_share"]):
        errors.append("complaint_type_share is not numeric")

    if errors:
        for e in errors:
            print(f"\n[VALIDATION ERROR] {e}")
        sys.exit(1)

    print("\n[ALL VALIDATIONS PASSED]")

    # 11. First 30 rows
    print("\nFirst 30 rows:")
    pd.set_option("display.max_colwidth", 60)
    print(df.head(30).to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

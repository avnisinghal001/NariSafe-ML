"""
Extract city-wise women crime data (2021-2023) from NCRB PDF.

Source: data/raw/crime_women_citywise_2021_2023.pdf  (Table 3B.1)
Output: data/intermediate/women_crime_citywise.csv
"""

import re
import sys
import pdfplumber
import pandas as pd

PDF_PATH = "data/raw/crime_women_citywise_2021_2023.pdf"
OUTPUT_PATH = "data/intermediate/women_crime_citywise.csv"

# Canonical city names used across all project files (city_coordinates.csv /
# osm_feature_summary.csv).  The PDF spells every city identically, so no
# remapping is needed — but the list is kept explicit so validation is strict.
EXPECTED_CITIES = [
    "Agra", "Amritsar", "Asansol", "Aurangabad", "Bhopal",
    "Chandigarh City", "Dhanbad", "Durg-Bhilainagar", "Faridabad", "Gwalior",
    "Jabalpur", "Jamshedpur", "Jodhpur", "Kannur", "Kollam",
    "Kota", "Ludhiana", "Madurai", "Malappuram", "Meerut",
    "Nasik", "Prayagraj", "Raipur", "Rajkot", "Ranchi",
    "Srinagar", "Thiruvananthapuram", "Thrissur", "Tiruchirapalli", "Vadodara",
    "Varanasi", "Vasai Virar", "Vijayawada", "Vishakhapatnam",
]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_table_rows(pdf_path: str) -> list[dict]:
    """
    Parse Table 3B.1 from the PDF.

    The PDF is a single-page text-layer PDF.  Each data row looks like:
        <SL>  <City name (may be multi-word)>  <2021>  <2022>  <2023>
        <pop_lakhs>  <crime_rate_2023>  <chargesheeting_rate_2023>

    The TOTAL row at the bottom is skipped.
    """
    rows = []

    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) != 1:
            # Warn but continue — future editions may add pages.
            print(f"[WARN] Expected 1 page, got {len(pdf.pages)}. "
                  "Processing all pages.")

        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            for line in text.splitlines():
                line = line.strip()
                # Match lines that start with a 1- or 2-digit serial number
                # followed by a city name and then 6 numeric fields.
                # Pattern: <SL> <city words> <int> <int> <int> <float> <float> <float>
                m = re.match(
                    r'^(\d{1,2})\s+'           # [1] serial number
                    r'([\w\s\-]+?)\s+'          # [2] city name (lazy)
                    r'(\d+)\s+'                 # [3] 2021 crimes
                    r'(\d+)\s+'                 # [4] 2022 crimes
                    r'(\d+)\s+'                 # [5] 2023 crimes
                    r'([\d.]+)\s+'              # [6] population (lakhs)
                    r'([\d.]+)\s+'              # [7] crime rate 2023
                    r'([\d.]+)$',               # [8] chargesheeting rate 2023
                    line,
                )
                if m:
                    city_raw = m.group(2).strip()
                    # Skip the aggregated TOTAL row
                    if city_raw.upper().startswith("TOTAL"):
                        continue
                    rows.append({
                        "city": city_raw,
                        "women_crime_2021": int(m.group(3)),
                        "women_crime_2022": int(m.group(4)),
                        "women_crime_2023": int(m.group(5)),
                        "population_lakhs": float(m.group(6)),
                        "women_crime_rate_2023": float(m.group(7)),
                        "chargesheeting_rate_2023": float(m.group(8)),
                    })

    return rows


# ---------------------------------------------------------------------------
# City-name normalisation
# ---------------------------------------------------------------------------

# The PDF uses identical spellings to the project's canonical list, so no
# remapping is required.  This dict is kept as an explicit safety net
CITY_NAME_MAP: dict[str, str] = {}


def normalise_city(name: str) -> str:
    return CITY_NAME_MAP.get(name, name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Reading PDF: {PDF_PATH}")
    rows = extract_table_rows(PDF_PATH)

    if not rows:
        sys.exit("[ERROR] No rows extracted from PDF. Check the regex or PDF layout.")

    df = pd.DataFrame(rows)

    # Normalise city names
    df["city"] = df["city"].map(normalise_city)

    # -----------------------------------------------------------------------
    # Calculate derived column
    # -----------------------------------------------------------------------
    df["women_crime_growth_21_23"] = (
        (df["women_crime_2023"] - df["women_crime_2021"]) / df["women_crime_2021"]
    ).round(4)

    # Reorder columns to spec
    df = df[[
        "city",
        "women_crime_2021", "women_crime_2022", "women_crime_2023",
        "population_lakhs",
        "women_crime_rate_2023",
        "chargesheeting_rate_2023",
        "women_crime_growth_21_23",
    ]]

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------
    errors = []

    # 1. Exactly 34 rows
    if len(df) != 34:
        errors.append(f"Row count: expected 34, got {len(df)}")

    # 2. No missing city names
    missing_names = df["city"].isna() | (df["city"] == "")
    if missing_names.any():
        errors.append(f"Missing city names at rows: {df.index[missing_names].tolist()}")

    # 3. No duplicate cities
    dupes = df["city"][df["city"].duplicated()].tolist()
    if dupes:
        errors.append(f"Duplicate cities: {dupes}")

    # 4. All extracted cities match the canonical list
    extracted = set(df["city"])
    expected = set(EXPECTED_CITIES)
    extra = extracted - expected
    missing = expected - extracted
    if extra:
        errors.append(f"Unexpected cities (not in canonical list): {sorted(extra)}")
    if missing:
        errors.append(f"Missing cities (in canonical list but not extracted): {sorted(missing)}")

    # 5. Numeric columns are numeric
    numeric_cols = [
        "women_crime_2021", "women_crime_2022", "women_crime_2023",
        "population_lakhs", "women_crime_rate_2023",
        "chargesheeting_rate_2023", "women_crime_growth_21_23",
    ]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' is not numeric")

    if errors:
        for e in errors:
            print(f"[VALIDATION ERROR] {e}")
        sys.exit(1)

    print("\n[VALIDATION PASSED]")

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------
    print(f"\nCity count: {len(df)}")

    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

    print("\nTop 10 cities by women_crime_rate_2023:")
    top10 = df.nlargest(10, "women_crime_rate_2023")[
        ["city", "women_crime_rate_2023", "women_crime_2023"]
    ]
    print(top10.to_string(index=False))

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

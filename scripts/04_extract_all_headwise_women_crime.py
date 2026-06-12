"""
Extract city-wise, crime-head-wise women crime data (2023) from NCRB PDF.

Source : data/raw/crime_women_headwise_citywise_2023.pdf  (Table 3B.2, 16 pages)
Ref    : data/intermediate/women_crime_citywise.csv       (women_crime_2023 per city)
Output : data/intermediate/women_crime_headwise_all_types.csv

Only the I (Incidences/Cases) column is extracted from each IVR triplet.
Parsing strategy: for each data row, re.findall extracts all numbers from the
line.  The first integer is the SL number (1-34); the remaining values are
exactly n_types*3 numbers (I, V, R for each type on that page).  Any line that
does not satisfy this count is silently skipped (handles header rows, footnotes,
totals, and page-number lines without special-casing).
"""

import re
import sys
import pdfplumber
import pandas as pd

PDF_PATH = "data/raw/crime_women_headwise_citywise_2023.pdf"
CITYWISE_PATH = "data/intermediate/women_crime_citywise.csv"
OUTPUT_PATH = "data/intermediate/women_crime_headwise_all_types.csv"

# Canonical 34 cities ordered by SL number as they appear in the PDF.
# Matches city_coordinates.csv and osm_feature_summary.csv exactly.
EXPECTED_CITIES = [
    "Agra", "Amritsar", "Asansol", "Aurangabad", "Bhopal",
    "Chandigarh City", "Dhanbad", "Durg-Bhilainagar", "Faridabad", "Gwalior",
    "Jabalpur", "Jamshedpur", "Jodhpur", "Kannur", "Kollam",
    "Kota", "Ludhiana", "Madurai", "Malappuram", "Meerut",
    "Nasik", "Prayagraj", "Raipur", "Rajkot", "Ranchi",
    "Srinagar", "Thiruvananthapuram", "Thrissur", "Tiruchirapalli", "Vadodara",
    "Varanasi", "Vasai Virar", "Vijayawada", "Vishakhapatnam",
]

# ──────────────────────────────────────────────────────────────────────────────
# Crime-type catalog: (page_number, position_on_page, original_name, clean_name)
#
# position_on_page is the 0-based index of the crime type's IVR triplet within
# its page.  Column numbers from the PDF are noted in the comment per block.
# ──────────────────────────────────────────────────────────────────────────────
CRIME_TYPES = [
    # Page 1 — cols [3-11] — IPC misc. ----------------------------------------
    (1,  0, "Murder with Rape/Gang Rape IPC",
            "murder_with_rape_gang_rape_ipc"),
    (1,  1, "Dowry Deaths (Sec. 304B IPC)",
            "dowry_deaths"),
    (1,  2, "Abetment to Suicide of Women (Sec. 305/306 IPC)",
            "abetment_to_suicide_of_women"),

    # Page 2 — cols [12-23] — IPC misc. (continued) ---------------------------
    (2,  0, "Miscarriage (Sec. 313 & 314 IPC)",
            "miscarriage"),
    (2,  1, "Acid Attack (Sec. 326A IPC)",
            "acid_attack"),
    (2,  2, "Attempt to Acid Attack (Sec. 326B IPC)",
            "attempt_to_acid_attack"),
    (2,  3, "Cruelty by Husband or his Relatives (Sec. 498A IPC)",
            "cruelty_by_husband_or_relatives"),

    # Page 3 — cols [24-32] — K&A totals --------------------------------------
    (3,  0, "Kidnapping and Abduction of Women (Total)",
            "kidnapping_and_abduction_of_women"),
    (3,  1, "Kidnapping and Abduction (Sec. 363 IPC)",
            "kidnapping_and_abduction_sec_363"),
    (3,  2, "Kidnapping and Abduction in order to Murder (Sec. 364 IPC)",
            "kidnapping_and_abduction_to_murder"),

    # Page 4 — cols [33-41] — K&A sub-types -----------------------------------
    (4,  0, "Kidnapping for Ransom (Sec. 364A IPC)",
            "kidnapping_for_ransom"),
    (4,  1, "K&A to Compel for Marriage - Total (Sec. 366 IPC)",
            "kna_to_compel_for_marriage_total"),
    (4,  2, "K&A for Marriage - Women Above 18 Yrs (Sec. 366 IPC)",
            "kna_for_marriage_women_above_18"),

    # Page 5 — cols [42-50] — K&A sub-types (continued) ----------------------
    (5,  0, "K&A for Marriage - Girls Below 18 Yrs (Sec. 366 IPC)",
            "kna_for_marriage_girls_below_18"),
    (5,  1, "Importation of Girls from Foreign Country (Sec. 366B IPC)",
            "importation_of_girls_foreign_country"),
    (5,  2, "Procuration of Minor Girls (Sec. 366A IPC)",
            "procuration_of_minor_girls"),

    # Page 6 — cols [51-59] — K&A others + trafficking -----------------------
    (6,  0, "Kidnapping and Abduction of Women - Others",
            "kidnapping_abduction_others"),
    (6,  1, "Human Trafficking (Sec. 370 & 370A IPC)",
            "human_trafficking"),
    (6,  2, "Selling of Minor Girls (Sec. 372 IPC)",
            "selling_of_minor_girls"),

    # Page 7 — cols [60-71] — buying + rape -----------------------------------
    (7,  0, "Buying of Minor Girls (Sec. 373 IPC)",
            "buying_of_minor_girls"),
    (7,  1, "Rape (Total)",
            "rape_total"),
    (7,  2, "Rape - Women 18 Yrs and Above (Sec. 376 IPC)",
            "rape_women_18_and_above"),
    (7,  3, "Rape - Girls Below 18 Yrs (Sec. 376 IPC)",
            "rape_girls_below_18"),

    # Page 8 — cols [72-80] — attempt to commit rape --------------------------
    (8,  0, "Attempt to Commit Rape (Total)",
            "attempt_to_commit_rape_total"),
    (8,  1, "Attempt to Commit Rape - Women 18 Yrs and Above",
            "attempt_rape_women_18_and_above"),
    (8,  2, "Attempt to Commit Rape - Girls Below 18 Yrs",
            "attempt_rape_girls_below_18"),

    # Page 9 — cols [81-89] — assault on women --------------------------------
    (9,  0, "Assault on Women with Intent to Outrage her Modesty (Total)",
            "assault_on_women_with_intent_to_outrage_modesty"),
    (9,  1, "Assault on Women - Women 18 Yrs and Above (Sec. 354 IPC)",
            "assault_women_18_and_above"),
    (9,  2, "Assault on Women - Girls Below 18 Yrs (Sec. 354 IPC)",
            "assault_girls_below_18"),

    # Page 10 — cols [90-101] — insult + IPC total ----------------------------
    (10, 0, "Insult to the Modesty of Women (Total)",
            "insult_to_modesty_of_women"),
    (10, 1, "Insult to Modesty - Women 18 Yrs and Above (Sec. 509 IPC)",
            "insult_modesty_women_18_and_above"),
    (10, 2, "Insult to Modesty - Girls Below 18 Yrs (Sec. 509 IPC)",
            "insult_modesty_girls_below_18"),
    (10, 3, "Total IPC Crimes against Women",
            "total_ipc_crimes_against_women"),

    # Page 11 — cols [102-113] — SLL: Dowry Prohibition + ITP start ----------
    (11, 0, "Dowry Prohibition Act 1961",
            "dowry_prohibition_act"),
    (11, 1, "Immoral Traffic (Prevention) Act 1956 (Total)",
            "immoral_traffic_prevention_act_total"),
    (11, 2, "ITP Act - Procuring/Inducing Children for Prostitution (Sec. 5)",
            "itp_procuring_inducing_children"),
    (11, 3, "ITP Act - Detaining in Premises Where Prostitution Carried On (Sec. 6)",
            "itp_detaining_in_premises"),

    # Page 12 — cols [114-125] — SLL: ITP continued + DV Act -----------------
    (12, 0, "ITP Act - Prostitution in or Near Public Places (Sec. 7)",
            "itp_prostitution_near_public_places"),
    (12, 1, "ITP Act - Seducing or Soliciting for Prostitution (Sec. 8)",
            "itp_seducing_soliciting"),
    (12, 2, "ITP Act - Other Sections",
            "itp_other_sections"),
    (12, 3, "Protection of Women from Domestic Violence Act",
            "domestic_violence_act"),

    # Page 13 — cols [126-134] — SLL: Cyber Crimes ---------------------------
    (13, 0, "Cyber Crimes Against Women (Total)",
            "cyber_crimes_against_women_total"),
    (13, 1, "Cyber Crimes - Publishing Sexually Explicit Material (Sec. 67A/67B IT Act)",
            "cyber_publishing_explicit_material"),
    (13, 2, "Cyber Crimes - Other Women Centric Cyber Crimes",
            "cyber_other_women_centric"),

    # Page 14 — cols [135-146] — SLL: POCSO (start) --------------------------
    (14, 0, "Protection of Children from Sexual Offences Act (Total)",
            "pocso_act_total"),
    (14, 1, "POCSO - Child Rape (Sec. 4 & 6 of POCSO / Sec. 376 IPC)",
            "pocso_child_rape"),
    (14, 2, "POCSO - Sexual Assault of Children (Sec. 8 & 10 / Sec. 354 IPC)",
            "pocso_sexual_assault"),
    (14, 3, "POCSO - Sexual Harassment (Sec. 12 / Sec. 509 IPC)",
            "pocso_sexual_harassment"),

    # Page 15 — cols [147-155] — SLL: POCSO (continued) ----------------------
    (15, 0, "POCSO - Child Pornography (Sec. 14 & 15 of POCSO)",
            "pocso_child_pornography"),
    (15, 1, "POCSO - Other Offences (Sec. 17 to 22 of POCSO)",
            "pocso_other_offences"),
    (15, 2, "POCSO r/w Section 377 IPC / Unnatural Offences",
            "pocso_unnatural_offences"),

    # Page 16 — cols [156-164] — SLL: remaining + grand totals ---------------
    (16, 0, "Indecent Representation of Women (Prohibition) Act 1986",
            "indecent_representation_of_women_act"),
    (16, 1, "Total SLL Crimes against Women",
            "total_sll_crimes_against_women"),
    (16, 2, "Total Crime against Women (IPC+SLL)",
            "total_crime_against_women"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def _page_type_map() -> dict:
    """Build {page_number: [(position, original, clean), ...]} from CRIME_TYPES."""
    result: dict = {}
    for pg, pos, orig, clean in CRIME_TYPES:
        result.setdefault(pg, []).append((pos, orig, clean))
    return result


def extract_page_data(text: str, n_types: int, page_num: int) -> dict:
    """
    Parse one page and return {sl_number: [i_val_0, i_val_1, ...]} for each
    data row (SL 1-34).

    Each data line yields exactly n_types*3 numbers after the SL integer.
    Any line that fails this count is silently skipped — this handles header
    rows, totals, footnotes, and page-number lines without special-casing.
    """
    expected = n_types * 3
    results: dict = {}

    for line in text.splitlines():
        nums = re.findall(r'\d+(?:\.\d+)?', line)
        if not nums:
            continue

        first = nums[0]
        if '.' in first:          # SL is always a plain integer
            continue
        sl = int(first)
        if not (1 <= sl <= 34):
            continue
        if sl in results:          # keep only the first match per SL per page
            continue

        remaining = nums[1:]
        if len(remaining) != expected:
            continue

        # I column is the first of each IVR triplet
        i_values = [float(remaining[i * 3]) for i in range(n_types)]
        results[sl] = i_values

    missing = set(range(1, 35)) - set(results.keys())
    if missing:
        print(f"  [WARN] Page {page_num}: missing SL numbers {sorted(missing)}")

    return results


def build_crime_matrix() -> dict:
    """
    Open the PDF and populate {sl_number: {crime_type_clean: count}}.
    sl_number runs 1-34, matching EXPECTED_CITIES index offset by 1.
    """
    page_type_map = _page_type_map()
    matrix = {sl: {} for sl in range(1, 35)}

    with pdfplumber.open(PDF_PATH) as pdf:
        for pg_num in range(1, len(pdf.pages) + 1):
            if pg_num not in page_type_map:
                continue

            text = pdf.pages[pg_num - 1].extract_text()
            if not text:
                print(f"  [ERROR] Page {pg_num}: no text extracted")
                continue

            types_on_page = page_type_map[pg_num]
            n_types = max(pos for pos, _, _ in types_on_page) + 1
            page_data = extract_page_data(text, n_types, pg_num)

            for sl, i_values in page_data.items():
                for pos, orig, clean in types_on_page:
                    matrix[sl][clean] = int(i_values[pos])

    return matrix


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Reading PDF: {PDF_PATH}")
    matrix = build_crime_matrix()

    clean_to_orig = {clean: orig for _, _, orig, clean in CRIME_TYPES}
    ordered_clean = [clean for _, _, _, clean in CRIME_TYPES]

    # Build long-format DataFrame
    records = []
    for sl, city in enumerate(EXPECTED_CITIES, start=1):
        for clean in ordered_clean:
            records.append({
                "city": city,
                "complaint_type_original": clean_to_orig[clean],
                "complaint_type_clean": clean,
                "complaint_type_count": matrix[sl].get(clean),
            })

    df = pd.DataFrame(records)
    df["complaint_type_count"] = pd.to_numeric(df["complaint_type_count"])

    # Merge women_crime_2023 for share calculation
    citywise = pd.read_csv(CITYWISE_PATH)[["city", "women_crime_2023"]]
    df = df.merge(citywise, on="city", how="left")
    df["complaint_type_share"] = (
        df["complaint_type_count"] / df["women_crime_2023"]
    ).round(6)
    df = df.drop(columns=["women_crime_2023"])

    # ── Cross-validation ──────────────────────────────────────────────────────
    # total_crime_against_women (col 162 in PDF) must equal women_crime_2023
    # from the earlier script.  A mismatch means a page was parsed incorrectly.
    totals = (
        df[df["complaint_type_clean"] == "total_crime_against_women"]
        .set_index("city")["complaint_type_count"]
    )
    ref = citywise.set_index("city")["women_crime_2023"]
    mismatches = {
        city: (int(totals[city]), int(ref[city]))
        for city in EXPECTED_CITIES
        if int(totals[city]) != int(ref[city])
    }
    if mismatches:
        print(f"\n[WARN] {len(mismatches)} cross-validation mismatches "
              "(total_crime_against_women vs women_crime_2023):")
        for city, (got, want) in mismatches.items():
            print(f"  {city}: extracted={got}, reference={want}")
    else:
        print("\n[CROSS-VALIDATION PASSED] "
              "total_crime_against_women == women_crime_2023 for all 34 cities")

    # ── Validation ────────────────────────────────────────────────────────────
    errors = []

    # No duplicate city + complaint_type_clean pairs
    dupes = df[df.duplicated(["city", "complaint_type_clean"])]
    if not dupes.empty:
        errors.append(
            f"Duplicate city+type pairs: "
            f"{dupes[['city','complaint_type_clean']].values.tolist()}"
        )

    # Every city has the same full set of complaint types
    all_types_set = set(ordered_clean)
    for city, grp in df.groupby("city"):
        city_types = set(grp["complaint_type_clean"])
        missing = all_types_set - city_types
        extra = city_types - all_types_set
        if missing or extra:
            errors.append(f"{city}: missing={missing}, extra={extra}")

    # Numeric columns are numeric
    for col in ("complaint_type_count", "complaint_type_share"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' is not numeric")

    if errors:
        for e in errors:
            print(f"[VALIDATION ERROR] {e}")
        sys.exit(1)

    print("[VALIDATION PASSED]")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print(f"\nTotal rows          : {len(df)}")
    print(f"Unique cities       : {df['city'].nunique()}")
    print(f"Unique complaint types: {df['complaint_type_clean'].nunique()}")

    print("\nComplaint types list:")
    for ct in df["complaint_type_original"].unique():
        print(f"  - {ct}")

    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    print("\nFirst 30 rows:")
    pd.set_option("display.max_colwidth", 60)
    print(df.head(30).to_string(index=False))

    print("\nTop 10 complaint types by total count across all cities:")
    top10 = (
        df[~df["complaint_type_clean"].str.startswith("total_")]
        .groupby("complaint_type_original")["complaint_type_count"]
        .sum()
        .nlargest(10)
        .reset_index()
    )
    print(top10.to_string(index=False))

    print("\nTop 5 cities for each major complaint type:")
    major_types = [
        "cruelty_by_husband_or_relatives",
        "kidnapping_and_abduction_of_women",
        "rape_total",
        "assault_on_women_with_intent_to_outrage_modesty",
        "pocso_act_total",
        "domestic_violence_act",
        "cyber_crimes_against_women_total",
        "dowry_deaths",
    ]
    for ct in major_types:
        subset = (
            df[df["complaint_type_clean"] == ct]
            .nlargest(5, "complaint_type_count")[["city", "complaint_type_count"]]
        )
        print(f"\n  {clean_to_orig[ct]}:")
        print(subset.to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

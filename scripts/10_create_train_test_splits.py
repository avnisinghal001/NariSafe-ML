import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit


INPUT_PATH = Path("data/processed/ml_ready/narisafe_ml_features.csv")
CONFIG_PATH = Path("data/processed/ml_ready/feature_config.json")

OUTPUT_DIR = Path("data/processed/ml_ready/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20


def print_split_report(name, df):
    print(f"\n{name}")
    print("-" * 60)
    print("Shape:", df.shape)

    print("\nRisk level distribution:")
    print(df["risk_level"].value_counts())

    print("\nRisk level percentage:")
    print((df["risk_level"].value_counts(normalize=True) * 100).round(2))

    print("\nUnique cities:", df["city"].nunique())
    print(sorted(df["city"].unique()))


def save_split_report(report_path, split_info):
    with open(report_path, "w") as f:
        f.write("# NariSafe Train/Test Split Report\n\n")

        for section, info in split_info.items():
            f.write(f"## {section}\n\n")
            f.write(f"Shape: `{info['shape']}`\n\n")
            f.write(f"Unique cities: `{info['unique_cities']}`\n\n")

            f.write("Cities:\n\n")
            for city in info["cities"]:
                f.write(f"- {city}\n")

            f.write("\nRisk distribution:\n\n")
            for label, count in info["risk_counts"].items():
                f.write(f"- {label}: {count}\n")

            f.write("\nRisk percentage:\n\n")
            for label, pct in info["risk_percentages"].items():
                f.write(f"- {label}: {pct}%\n")

            f.write("\n")


def collect_info(df):
    return {
        "shape": df.shape,
        "unique_cities": df["city"].nunique(),
        "cities": sorted(df["city"].unique().tolist()),
        "risk_counts": df["risk_level"].value_counts().to_dict(),
        "risk_percentages": (df["risk_level"].value_counts(normalize=True) * 100).round(2).to_dict(),
    }


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    df = pd.read_csv(INPUT_PATH)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    target_col = config["target_column"]

    print("\n================ TRAIN TEST SPLIT CREATION ================\n")

    print("Input shape:", df.shape)
    print("Target:", target_col)

    # -----------------------------------------------------
    # Split 1: Random stratified split
    # -----------------------------------------------------
    random_train_val, random_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[target_col]
    )

    random_train, random_val = train_test_split(
        random_train_val,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=random_train_val[target_col]
    )

    random_train.to_csv(OUTPUT_DIR / "random_train.csv", index=False)
    random_val.to_csv(OUTPUT_DIR / "random_val.csv", index=False)
    random_test.to_csv(OUTPUT_DIR / "random_test.csv", index=False)

    # -----------------------------------------------------
    # Split 2: Group split by city
    # -----------------------------------------------------
    group_split = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    groups = df["city"]

    train_val_idx, group_test_idx = next(
        group_split.split(df, df[target_col], groups=groups)
    )

    group_train_val = df.iloc[train_val_idx].copy()
    group_test = df.iloc[group_test_idx].copy()

    group_split_val = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )

    train_idx, val_idx = next(
        group_split_val.split(
            group_train_val,
            group_train_val[target_col],
            groups=group_train_val["city"]
        )
    )

    group_train = group_train_val.iloc[train_idx].copy()
    group_val = group_train_val.iloc[val_idx].copy()

    group_train.to_csv(OUTPUT_DIR / "group_city_train.csv", index=False)
    group_val.to_csv(OUTPUT_DIR / "group_city_val.csv", index=False)
    group_test.to_csv(OUTPUT_DIR / "group_city_test.csv", index=False)

    # -----------------------------------------------------
    # Print reports
    # -----------------------------------------------------
    print_split_report("Random Train", random_train)
    print_split_report("Random Validation", random_val)
    print_split_report("Random Test", random_test)

    print_split_report("Group City Train", group_train)
    print_split_report("Group City Validation", group_val)
    print_split_report("Group City Test", group_test)

    split_info = {
        "Random Train": collect_info(random_train),
        "Random Validation": collect_info(random_val),
        "Random Test": collect_info(random_test),
        "Group City Train": collect_info(group_train),
        "Group City Validation": collect_info(group_val),
        "Group City Test": collect_info(group_test),
    }

    save_split_report(
        OUTPUT_DIR / "split_report.md",
        split_info
    )

    print("\nSaved split files in:")
    print(OUTPUT_DIR)

    print("\nFiles created:")
    print("- random_train.csv")
    print("- random_val.csv")
    print("- random_test.csv")
    print("- group_city_train.csv")
    print("- group_city_val.csv")
    print("- group_city_test.csv")
    print("- split_report.md")

    print("\n================ SPLIT CREATION COMPLETE ================\n")


if __name__ == "__main__":
    main()
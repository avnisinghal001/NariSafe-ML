from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit


ORIGINAL_DATA = Path("data/processed/ml_ready/narisafe_ml_features.csv")
NO_LOCATION_DATA = Path("data/processed/ml_ready_no_location/narisafe_ml_features_no_location.csv")

OUTPUT_DIR = Path("data/processed/ml_ready_no_location/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
TARGET = "risk_level"


def main():
    original_df = pd.read_csv(ORIGINAL_DATA)
    df = pd.read_csv(NO_LOCATION_DATA)

    # We need city only for grouping, but it will not be saved as a model feature.
    groups = original_df["city"]

    print("\n================ NO-LOCATION SPLITS ================\n")
    print("No-location input shape:", df.shape)

    # Random split
    random_train_val, random_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )

    random_train, random_val = train_test_split(
        random_train_val,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=random_train_val[TARGET],
    )

    random_train.to_csv(OUTPUT_DIR / "random_train.csv", index=False)
    random_val.to_csv(OUTPUT_DIR / "random_val.csv", index=False)
    random_test.to_csv(OUTPUT_DIR / "random_test.csv", index=False)

    # Group-city split using city from original dataset
    group_split = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_val_idx, test_idx = next(
        group_split.split(df, df[TARGET], groups=groups)
    )

    group_train_val = df.iloc[train_val_idx].copy()
    group_test = df.iloc[test_idx].copy()

    group_train_val_groups = groups.iloc[train_val_idx].reset_index(drop=True)

    group_split_val = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, val_idx = next(
        group_split_val.split(
            group_train_val,
            group_train_val[TARGET],
            groups=group_train_val_groups,
        )
    )

    group_train = group_train_val.iloc[train_idx].copy()
    group_val = group_train_val.iloc[val_idx].copy()

    group_train.to_csv(OUTPUT_DIR / "group_city_train.csv", index=False)
    group_val.to_csv(OUTPUT_DIR / "group_city_val.csv", index=False)
    group_test.to_csv(OUTPUT_DIR / "group_city_test.csv", index=False)

    print("Random train:", random_train.shape)
    print("Random val:", random_val.shape)
    print("Random test:", random_test.shape)

    print("\nGroup train:", group_train.shape)
    print("Group val:", group_val.shape)
    print("Group test:", group_test.shape)

    print("\nSaved splits to:")
    print(OUTPUT_DIR)

    print("\n================ COMPLETE ================\n")


if __name__ == "__main__":
    main()
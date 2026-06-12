import json
from pathlib import Path

import pandas as pd


INPUT_DATA = Path("data/processed/ml_ready/narisafe_ml_features.csv")
INPUT_CONFIG = Path("data/processed/ml_ready/feature_config.json")

OUTPUT_DIR = Path("data/processed/ml_ready_no_location")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DATA = OUTPUT_DIR / "narisafe_ml_features_no_location.csv"
OUTPUT_CONFIG = OUTPUT_DIR / "feature_config_no_location.json"


DROP_FEATURES = [
    "city",
    "latitude",
    "longitude",
]


def main():
    df = pd.read_csv(INPUT_DATA)

    with open(INPUT_CONFIG, "r") as f:
        config = json.load(f)

    categorical_features = config["categorical_features"]
    numeric_features = config["numeric_features"]
    target_column = config["target_column"]

    new_categorical_features = [
        col for col in categorical_features
        if col not in DROP_FEATURES
    ]

    new_numeric_features = [
        col for col in numeric_features
        if col not in DROP_FEATURES
    ]

    final_cols = new_categorical_features + new_numeric_features + [target_column]
    final_df = df[final_cols].copy()

    new_config = {
        "target_column": target_column,
        "categorical_features": new_categorical_features,
        "numeric_features": new_numeric_features,
        "dropped_features": DROP_FEATURES,
        "notes": [
            "This dataset removes city, latitude, and longitude.",
            "Purpose: test whether the model can learn general contextual safety patterns without direct city identity.",
            "Use group-city split for honest evaluation."
        ]
    }

    final_df.to_csv(OUTPUT_DATA, index=False)

    with open(OUTPUT_CONFIG, "w") as f:
        json.dump(new_config, f, indent=2)

    print("\n================ NO-LOCATION FEATURE DATASET CREATED ================\n")
    print("Input shape:", df.shape)
    print("Output shape:", final_df.shape)

    print("\nDropped features:")
    for col in DROP_FEATURES:
        print("-", col)

    print("\nNew categorical features:")
    print(new_categorical_features)

    print("\nNew numeric features:")
    print(new_numeric_features)

    print("\nSaved:")
    print(OUTPUT_DATA)
    print(OUTPUT_CONFIG)


if __name__ == "__main__":
    main()
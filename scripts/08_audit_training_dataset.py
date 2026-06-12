import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/narisafe_model_training_dataset.csv")
OUTPUT_DIR = Path("data/processed/audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"File not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    print("\nDATASET AUDIT\n")

    print("Shape:")
    print(df.shape)

    print("\nColumns:")
    for col in df.columns:
        print("-", col)

    print("\nTarget distribution:")
    print(df["risk_level"].value_counts())
    print("\nTarget distribution percentage:")
    print((df["risk_level"].value_counts(normalize=True) * 100).round(2))

    print("\nMissing values:")
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing[missing > 0])

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nUnique values in categorical columns:")
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in categorical_cols:
        print(f"\n{col}:")
        print("unique count:", df[col].nunique())
        print(df[col].value_counts().head(15))

    print("\nNumeric column summary:")
    numeric_summary = df.describe().T
    print(numeric_summary)

    # Save reports
    missing.to_csv(OUTPUT_DIR / "missing_values.csv")
    numeric_summary.to_csv(OUTPUT_DIR / "numeric_summary.csv")

    categorical_summary = []

    for col in categorical_cols:
        categorical_summary.append({
            "column": col,
            "unique_count": df[col].nunique(),
            "top_value": df[col].value_counts().index[0],
            "top_value_count": df[col].value_counts().iloc[0]
        })

    pd.DataFrame(categorical_summary).to_csv(
        OUTPUT_DIR / "categorical_summary.csv",
        index=False
    )

    print("\nAudit files saved in:")
    print(OUTPUT_DIR)

    print("\nAUDIT COMPLETE\n")


if __name__ == "__main__":
    main()
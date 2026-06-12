import json
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/baselines/best_baseline_model.joblib")
CONFIG_PATH = Path("data/processed/ml_ready/feature_config.json")
OUTPUT_DIR = Path("reports/ml")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "feature_importance.csv"
OUTPUT_MD = OUTPUT_DIR / "feature_importance_report.md"


def get_feature_names(preprocessor, numeric_features, categorical_features):
    feature_names = []

    # Numeric feature names
    feature_names.extend(numeric_features)

    # One-hot categorical feature names
    cat_pipeline = preprocessor.named_transformers_["cat"]
    onehot = cat_pipeline.named_steps["onehot"]

    cat_feature_names = onehot.get_feature_names_out(categorical_features)
    feature_names.extend(cat_feature_names.tolist())

    return feature_names


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]

    model_pipeline = joblib.load(MODEL_PATH)

    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "Selected best model does not support feature_importances_. "
            "Use RandomForestClassifier or another tree-based model."
        )

    feature_names = get_feature_names(
        preprocessor=preprocessor,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    importances = model.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError(
            f"Feature name count {len(feature_names)} does not match "
            f"importance count {len(importances)}"
        )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    )

    importance_df.to_csv(OUTPUT_CSV, index=False)

    top_30 = importance_df.head(30)

    lines = []
    lines.append("# NariSafe Feature Importance Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This report explains which features contributed most to the selected baseline model."
    )
    lines.append("")
    lines.append("## Important note")
    lines.append("")
    lines.append(
        "Feature importance does not prove real-world causation. "
        "It only shows which features the model used most while learning the proxy risk labels."
    )
    lines.append("")
    lines.append("## Top 30 features")
    lines.append("")
    lines.append("| Rank | Feature | Importance |")
    lines.append("|---:|---|---:|")

    for rank, (_, row) in enumerate(top_30.iterrows(), start=1):
        lines.append(
            f"| {rank} | `{row['feature']}` | {row['importance']:.6f} |"
        )

    lines.append("")
    lines.append("## How to use this")
    lines.append("")
    lines.append(
        "Use this report in the README to explain that the model relies on "
        "crime rate, complaint type, severity, time, transport, police access, "
        "and urban context features instead of using the leakage column `risk_score`."
    )

    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines))

    print("\n================ FEATURE IMPORTANCE COMPLETE ================\n")
    print("Saved CSV:")
    print(OUTPUT_CSV)
    print("\nSaved report:")
    print(OUTPUT_MD)

    print("\nTop 30 features:")
    print(top_30)


if __name__ == "__main__":
    main()
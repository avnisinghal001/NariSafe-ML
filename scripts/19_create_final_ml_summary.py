import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd


MODEL_BUNDLE_PATH = Path("models/no_location_baselines/best_no_location_threshold_model.joblib")
THRESHOLD_CSV_PATH = Path("reports/ml_no_location/high_threshold_tuning.csv")
BASELINE_REPORT_PATH = Path("reports/ml_no_location/baseline_model_report.md")

OUTPUT_DIR = Path("reports/final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MD_PATH = OUTPUT_DIR / "final_ml_summary.md"
FINAL_JSON_PATH = OUTPUT_DIR / "final_model_manifest.json"


def load_model_bundle():
    bundle = joblib.load(MODEL_BUNDLE_PATH)

    if isinstance(bundle, dict):
        selected_threshold = (
            bundle.get("selected_threshold")
            or bundle.get("threshold")
            or 0.20
        )

        class_labels = (
            bundle.get("class_labels")
            or bundle.get("classes")
            or ["high", "low", "medium"]
        )
    else:
        selected_threshold = 0.20
        class_labels = ["high", "low", "medium"]

    return float(selected_threshold), list(class_labels)


def main():
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Model bundle not found: {MODEL_BUNDLE_PATH}")

    if not THRESHOLD_CSV_PATH.exists():
        raise FileNotFoundError(f"Threshold tuning CSV not found: {THRESHOLD_CSV_PATH}")

    selected_threshold, class_labels = load_model_bundle()

    threshold_df = pd.read_csv(THRESHOLD_CSV_PATH)

    # Best row according to selected threshold
    selected_row = threshold_df[
        threshold_df["threshold"].round(4) == round(selected_threshold, 4)
    ]

    if len(selected_row) == 0:
        selected_metrics = {}
    else:
        selected_metrics = selected_row.iloc[0].to_dict()

    manifest = {
        "project": "NariSafe",
        "created_at": datetime.now().isoformat(),
        "final_model_name": "No-location HistGradientBoosting with high-risk threshold tuning",
        "model_bundle_path": str(MODEL_BUNDLE_PATH),
        "model_type": "HistGradientBoostingClassifier",
        "prediction_type": "risk awareness level",
        "target": "risk_level",
        "classes": class_labels,
        "selected_high_risk_threshold": selected_threshold,
        "removed_features": [
            "city",
            "latitude",
            "longitude"
        ],
        "why_removed_features": (
            "Removed direct location identity to test whether the model can learn "
            "general contextual risk patterns instead of memorizing cities."
        ),
        "primary_evaluation_split": "group_city",
        "why_group_city_split": (
            "Group-city split tests generalization to unseen cities and is more honest "
            "than random split for this synthetic-expanded dataset."
        ),
        "final_metrics_from_threshold_tuning_csv": selected_metrics,
        "important_limitations": [
            "risk_level is a rule-based proxy label, not real incident-level ground truth.",
            "The model is a risk-awareness prototype, not a guaranteed crime prediction system.",
            "NCRB/OpenCity crime data is city-level, not hyperlocal incident-level data.",
            "OSM infrastructure features may be incomplete, especially street-light data.",
            "Prediction probabilities should be treated as model confidence, not real-world probability of crime."
        ]
    }

    with open(FINAL_JSON_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    lines = []

    lines.append("# NariSafe Final ML Summary")
    lines.append("")
    lines.append("## Final selected model")
    lines.append("")
    lines.append("**No-location HistGradientBoosting with high-risk threshold tuning**")
    lines.append("")
    lines.append("## Why this model was selected")
    lines.append("")
    lines.append(
        "This model was selected because it performed best on the realistic group-city evaluation setup "
        "after removing direct location identity features like city, latitude, and longitude."
    )
    lines.append("")
    lines.append("The final model uses contextual, crime, and infrastructure features instead of direct city identity.")
    lines.append("")

    lines.append("## Final model path")
    lines.append("")
    lines.append(f"`{MODEL_BUNDLE_PATH}`")
    lines.append("")

    lines.append("## Selected high-risk threshold")
    lines.append("")
    lines.append(f"`{selected_threshold}`")
    lines.append("")

    lines.append("## Removed features")
    lines.append("")
    lines.append("- `city`")
    lines.append("- `latitude`")
    lines.append("- `longitude`")
    lines.append("")

    lines.append("## Reason for removing location identity")
    lines.append("")
    lines.append(
        "City, latitude, and longitude were removed to reduce direct location memorization and test whether "
        "the model can learn general safety-context patterns from time, area type, complaint severity, "
        "crime statistics, police access, transport access, lighting, and crowd context."
    )
    lines.append("")

    lines.append("## Primary evaluation split")
    lines.append("")
    lines.append("`group_city`")
    lines.append("")
    lines.append(
        "Group-city split is treated as the main benchmark because it tests the model on cities not seen during training."
    )
    lines.append("")

    lines.append("## Final threshold-tuned metrics")
    lines.append("")
    if selected_metrics:
        for key, value in selected_metrics.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("Selected threshold metrics were not found in the tuning CSV.")
    lines.append("")

    lines.append("## Important limitations")
    lines.append("")
    lines.append("- `risk_level` is a rule-based proxy label, not real incident-level ground truth.")
    lines.append("- This is a risk-awareness prototype, not a guaranteed crime prediction system.")
    lines.append("- NCRB/OpenCity crime data is city-level, not hyperlocal incident-level data.")
    lines.append("- OSM infrastructure features may be incomplete, especially street-light data.")
    lines.append("- Prediction probabilities should be treated as model confidence, not real-world probability of crime.")
    lines.append("")

    lines.append("## Final recommendation")
    lines.append("")
    lines.append(
        "Use the threshold-tuned no-location model for the final demo and API. "
        "Display predictions as contextual risk-awareness levels, not as exact crime probabilities."
    )

    with open(FINAL_MD_PATH, "w") as f:
        f.write("\n".join(lines))

    print("\n================ FINAL ML SUMMARY CREATED ================\n")
    print("Saved final markdown summary:")
    print(FINAL_MD_PATH)

    print("\nSaved final model manifest:")
    print(FINAL_JSON_PATH)

    print("\nFinal model:")
    print(manifest["final_model_name"])

    print("\nSelected threshold:")
    print(selected_threshold)

    print("\n================ COMPLETE ================\n")


if __name__ == "__main__":
    main()
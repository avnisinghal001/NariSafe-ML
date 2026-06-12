import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_BUNDLE_PATH = Path("models/no_location_baselines/best_no_location_threshold_model.joblib")
CONFIG_PATH = Path("data/processed/ml_ready_no_location/feature_config_no_location.json")

# Use original full ML feature file for lookup because it still has city.
# But for model input, we will use no-location feature_config.
REFERENCE_DATA_PATH = Path("data/processed/ml_ready/narisafe_ml_features.csv")


def load_assets():
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Model bundle not found: {MODEL_BUNDLE_PATH}")

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(f"Reference data not found: {REFERENCE_DATA_PATH}")

    bundle = joblib.load(MODEL_BUNDLE_PATH)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    reference_df = pd.read_csv(REFERENCE_DATA_PATH)

    # Robust loading because bundle keys may vary depending on your script.
    if isinstance(bundle, dict):
        model = (
            bundle.get("pipeline")
            or bundle.get("model")
            or bundle.get("trained_pipeline")
        )

        selected_threshold = (
            bundle.get("selected_threshold")
            or bundle.get("threshold")
            or 0.20
        )

        class_labels = (
            bundle.get("class_labels")
            or bundle.get("classes")
            or None
        )
    else:
        model = bundle
        selected_threshold = 0.20
        class_labels = None

    if model is None:
        raise ValueError("Could not find trained model/pipeline inside the joblib bundle.")

    if class_labels is None:
        class_labels = list(model.classes_)

    return model, float(selected_threshold), list(class_labels), config, reference_df


def find_reference_row(
    reference_df,
    city,
    day_of_week,
    hour,
    area_context,
    complaint_type_clean,
):
    matched = reference_df[
        (reference_df["city"].str.lower() == city.lower())
        & (reference_df["day_of_week"].str.lower() == day_of_week.lower())
        & (reference_df["hour"] == hour)
        & (reference_df["area_context"].str.lower() == area_context.lower())
        & (reference_df["complaint_type_clean"].str.lower() == complaint_type_clean.lower())
    ]

    if len(matched) == 0:
        raise ValueError(
            "No matching row found. Check city, day_of_week, hour, "
            "area_context, and complaint_type_clean."
        )

    return matched.iloc[[0]].copy()


def threshold_predict(model, X, class_labels, high_threshold):
    probabilities = model.predict_proba(X)[0]

    probability_dict = {
        label: round(float(prob), 4)
        for label, prob in zip(class_labels, probabilities)
    }

    high_index = class_labels.index("high")
    high_probability = probabilities[high_index]

    if high_probability >= high_threshold:
        prediction = "high"
    else:
        # Choose between low and medium only.
        allowed_labels = ["low", "medium"]

        allowed_indices = [
            class_labels.index(label)
            for label in allowed_labels
            if label in class_labels
        ]

        best_index = allowed_indices[np.argmax(probabilities[allowed_indices])]
        prediction = class_labels[best_index]

    return prediction, probability_dict


def create_explanation(row):
    row = row.iloc[0]

    risk_factors = []
    mitigating_factors = []
    data_limitations = []

    # Time
    if row["time_bucket"] == "late_night":
        risk_factors.append("Late-night timing increases contextual risk.")
    elif row["time_bucket"] == "evening":
        risk_factors.append("Evening timing may increase contextual risk.")

    # Area
    if row["area_context"] in ["industrial", "transit"]:
        risk_factors.append(f"{row['area_context'].title()} area context can be more sensitive, especially during evening/night.")

    if row["area_context"] in ["market", "commercial"] and row["crowd_density"] == "high":
        mitigating_factors.append("High crowd density may increase natural surveillance.")

    if row["crowd_density"] == "low":
        risk_factors.append("Low crowd density may reduce natural surveillance.")

    # Complaint severity
    if row["complaint_severity"] == 3:
        risk_factors.append("Selected complaint type is high severity by crime category.")
    elif row["complaint_severity"] == 2:
        risk_factors.append("Selected complaint type is moderate severity by crime category.")

    # Complaint frequency
    if row["complaint_type_count"] == 0:
        data_limitations.append(
            "No recorded cases for this selected crime type in 2023 city-level NCRB data."
        )
    elif row["complaint_type_share"] >= 0.20:
        risk_factors.append(
            f"Selected complaint type has high city-level share: {row['complaint_type_share'] * 100:.1f}%."
        )

    # City crime rate
    if row["women_crime_rate_2023"] >= 200:
        risk_factors.append(
            f"City has high women crime rate: {row['women_crime_rate_2023']:.1f} per lakh population."
        )
    elif row["women_crime_rate_2023"] < 50:
        mitigating_factors.append(
            f"City has relatively low women crime rate: {row['women_crime_rate_2023']:.1f} per lakh population."
        )

    # Growth
    if row["women_crime_growth_21_23"] > 0.10:
        risk_factors.append(
            f"City crime trend increased by {row['women_crime_growth_21_23'] * 100:.1f}% from 2021 to 2023."
        )

    # Lighting
    if row["lighting_data_available"] == 0:
        data_limitations.append(
            "Street-light data is sparse or unavailable in OSM, so lighting context may be incomplete."
        )
    elif row["lighting_score"] <= 2:
        risk_factors.append("Lighting score is low.")

    # Police
    if row["police_access_score"] <= 1:
        risk_factors.append(
            f"Limited police access: nearest police station is {row['nearest_police_station_km']:.2f} km away."
        )

    # Transport
    if row["transport_access_score"] <= 1:
        risk_factors.append("Limited public transport availability nearby.")

    context = {
        "city": row["city"],
        "day_of_week": row["day_of_week"],
        "hour": int(row["hour"]),
        "time_bucket": row["time_bucket"],
        "area_context": row["area_context"],
        "complaint_type_clean": row["complaint_type_clean"],
        "complaint_severity_by_crime_type": int(row["complaint_severity"]),
        "complaint_type_count_2023": int(row["complaint_type_count"]),
        "complaint_type_share": round(float(row["complaint_type_share"]), 4),
        "women_crime_rate_2023": float(row["women_crime_rate_2023"]),
        "women_crime_growth_21_23": float(row["women_crime_growth_21_23"]),
        "lighting_score": int(row["lighting_score"]),
        "lighting_data_available": int(row["lighting_data_available"]),
        "crowd_density": row["crowd_density"],
        "nearest_police_station_km": round(float(row["nearest_police_station_km"]), 2),
        "police_access_score": int(row["police_access_score"]),
        "public_transport_count_3km": int(row["public_transport_count_3km"]),
        "transport_access_score": int(row["transport_access_score"]),
    }

    return context, risk_factors, mitigating_factors, data_limitations


def main():
    model, selected_threshold, class_labels, config, reference_df = load_assets()

    user_input = {
        "city": "Gwalior",
        "day_of_week": "Saturday",
        "hour": 23,
        "area_context": "transit",
        "complaint_type_clean": "assault_women_18_and_above",
    }

    input_row = find_reference_row(
        reference_df=reference_df,
        city=user_input["city"],
        day_of_week=user_input["day_of_week"],
        hour=user_input["hour"],
        area_context=user_input["area_context"],
        complaint_type_clean=user_input["complaint_type_clean"],
    )

    feature_cols = config["categorical_features"] + config["numeric_features"]
    X = input_row[feature_cols]

    prediction, probabilities = threshold_predict(
        model=model,
        X=X,
        class_labels=class_labels,
        high_threshold=selected_threshold,
    )

    context, risk_factors, mitigating_factors, data_limitations = create_explanation(input_row)

    print("\n================ NARISAFE THRESHOLD RISK PREDICTION ================\n")

    print("Input:")
    for key, value in user_input.items():
        print(f"{key}: {value}")

    print("\nSelected high-risk threshold:")
    print(selected_threshold)

    print("\nPredicted risk awareness level:")
    print(prediction)

    print("\nClass probabilities:")
    for label, prob in probabilities.items():
        print(f"{label}: {prob}")

    print("\nContext:")
    for key, value in context.items():
        print(f"{key}: {value}")

    print("\nKey risk factors:")
    for factor in risk_factors:
        print("-", factor)

    print("\nMitigating factors:")
    for factor in mitigating_factors:
        print("-", factor)

    print("\nData limitations:")
    for limitation in data_limitations:
        print("-", limitation)

    print("\nDisclaimer:")
    print(
        "This is a public-data-based risk-awareness prototype using proxy labels, "
        "not guaranteed real-world crime prediction."
    )

    print("\n================ COMPLETE ================\n")


if __name__ == "__main__":
    main()
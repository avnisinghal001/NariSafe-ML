import json
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/baselines/best_baseline_model.joblib")
CONFIG_PATH = Path("data/processed/ml_ready/feature_config.json")
REFERENCE_DATA_PATH = Path("data/processed/ml_ready/narisafe_ml_features.csv")


def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(f"Reference dataset not found: {REFERENCE_DATA_PATH}")

    model = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    reference_df = pd.read_csv(REFERENCE_DATA_PATH)

    return model, config, reference_df


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


def predict_risk(input_row, model, config):
    categorical_features = config["categorical_features"]
    numeric_features = config["numeric_features"]
    feature_cols = categorical_features + numeric_features

    X = input_row[feature_cols]

    prediction = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        classes = model.classes_

        probability_dict = {
            cls: round(float(prob), 4)
            for cls, prob in zip(classes, probabilities)
        }
    else:
        probability_dict = {}

    return prediction, probability_dict


def create_explanation(row):
    row = row.iloc[0]

    explanation = []

    explanation.append(f"City: {row['city']}")
    explanation.append(f"Day: {row['day_of_week']}")
    explanation.append(f"Hour: {row['hour']}")
    explanation.append(f"Time bucket: {row['time_bucket']}")
    explanation.append(f"Area context: {row['area_context']}")
    explanation.append(f"Complaint type: {row['complaint_type_clean']}")
    explanation.append(f"Complaint severity: {row['complaint_severity']}")
    explanation.append(f"Women crime rate 2023: {row['women_crime_rate_2023']}")
    explanation.append(f"Women crime growth 2021-2023: {row['women_crime_growth_21_23']}")
    explanation.append(f"Lighting score: {row['lighting_score']}")
    explanation.append(f"Lighting data available: {row['lighting_data_available']}")
    explanation.append(f"Crowd density: {row['crowd_density']}")
    explanation.append(f"Nearest police station km: {row['nearest_police_station_km']}")
    explanation.append(f"Police access score: {row['police_access_score']}")
    explanation.append(f"Public transport count 3km: {row['public_transport_count_3km']}")
    explanation.append(f"Transport access score: {row['transport_access_score']}")

    return explanation


def main():
    model, config, reference_df = load_assets()

    # Change this input for testing different scenarios.
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

    prediction, probabilities = predict_risk(
        input_row=input_row,
        model=model,
        config=config,
    )

    explanation = create_explanation(input_row)

    print("\n================ NARISAFE RISK PREDICTION ================\n")

    print("Input:")
    for key, value in user_input.items():
        print(f"{key}: {value}")

    print("\nPredicted risk level:")
    print(prediction)

    print("\nPrediction probabilities:")
    for label, prob in probabilities.items():
        print(f"{label}: {prob}")

    print("\nExplanation context:")
    for item in explanation:
        print("-", item)

    print("\nImportant note:")
    print(
        "This prediction is based on a rule-based proxy-label ML model. "
        "It should be used for risk awareness, not as guaranteed real-world crime prediction."
    )

    print("\n================ PREDICTION COMPLETE ================\n")


if __name__ == "__main__":
    main()
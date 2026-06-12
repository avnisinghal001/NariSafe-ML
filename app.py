import json

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download


# ============================================================
# Streamlit page config
# ============================================================

st.set_page_config(
    page_title="NariSafe Risk Awareness",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# Hugging Face repos
# ============================================================

HF_MODEL_REPO = "avnisinghal001/narisafe-risk-awareness-model"
HF_DATASET_REPO = "avnisinghal001/narisafe-risk-awareness-dataset"


# ============================================================
# Load model, dataset, config
# ============================================================

@st.cache_resource
def load_model_bundle():
    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename="best_no_location_threshold_model.joblib",
        repo_type="model",
    )
    return joblib.load(model_path)


@st.cache_data
def load_lookup_data():
    data_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename="narisafe_ml_features.csv",
        repo_type="dataset",
    )
    return pd.read_csv(data_path)


@st.cache_data
def load_feature_config():
    config_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename="feature_config_no_location.json",
        repo_type="model",
    )
    with open(config_path, "r") as f:
        return json.load(f)


# ============================================================
# Hidden complaint types for public demo
# ============================================================

PRIVATE_OR_NOT_DEMO_COMPLAINTS = {
    "cruelty_by_husband_or_relatives",
    "dowry_deaths",
    "dowry_prohibition_act",
    "domestic_violence_act",
    "abetment_to_suicide_of_women",

    "rape_girls_below_18",
    "attempt_rape_girls_below_18",
    "assault_girls_below_18",
    "insult_modesty_girls_below_18",
    "pocso_child_rape",
    "pocso_sexual_assault",
    "pocso_sexual_harassment",
    "pocso_child_pornography",
    "pocso_other_offences",
    "pocso_unnatural_offences",
}


# ============================================================
# Helper functions
# ============================================================

def pretty_label(value: str) -> str:
    return str(value).replace("_", " ").title()


def threshold_predict(pipeline, selected_threshold, feature_cols, row_df):
    x = row_df[feature_cols]

    probs = pipeline.predict_proba(x)[0]
    classes = list(pipeline.classes_)

    confidence = {cls: float(prob) for cls, prob in zip(classes, probs)}

    high_prob = confidence.get("high", 0.0)

    if high_prob >= selected_threshold:
        prediction = "high"
    else:
        low_prob = confidence.get("low", 0.0)
        medium_prob = confidence.get("medium", 0.0)
        prediction = "low" if low_prob >= medium_prob else "medium"

    return prediction, confidence


def show_risk_badge(level: str):
    if level == "high":
        st.error("High Risk Awareness Level")
    elif level == "medium":
        st.warning("Medium Risk Awareness Level")
    else:
        st.success("Low Risk Awareness Level")


def lighting_display(row):
    available = bool(int(row.get("lighting_data_available", 0)))
    score = int(row.get("lighting_score", 0))

    if not available:
        return "Unknown / estimated"

    return f"{score}/5"


def build_explanations(row):
    risk_factors = []
    mitigating_factors = []
    limitations = []

    time_bucket = str(row.get("time_bucket", ""))

    if time_bucket == "late_night":
        risk_factors.append("Late-night timing increases public-area safety concern.")
    elif time_bucket == "evening":
        risk_factors.append("Evening timing can increase movement and visibility-related risk.")
    else:
        mitigating_factors.append("Daytime timing is generally treated as lower contextual risk.")

    severity = int(row.get("complaint_severity", 1))
    if severity >= 3:
        risk_factors.append("Complaint type has high severity by crime type.")
    elif severity == 2:
        risk_factors.append("Complaint type has moderate severity by crime type.")
    else:
        mitigating_factors.append("Complaint type has lower severity by crime type.")

    complaint_share = float(row.get("complaint_type_share", 0))
    if complaint_share >= 0.20:
        risk_factors.append("This complaint type forms a high share of recorded women-crime cases in the city.")
    elif complaint_share >= 0.05:
        risk_factors.append("This complaint type has a noticeable share in the city's records.")
    else:
        mitigating_factors.append("This complaint type has a low share in the city's recorded cases.")

    crowd_density = str(row.get("crowd_density", ""))
    if crowd_density == "low":
        risk_factors.append("Low crowd density can reduce passive public visibility.")
    elif crowd_density == "high":
        mitigating_factors.append("High crowd density may provide better passive visibility.")

    crime_rate = float(row.get("women_crime_rate_2023", 0))
    if crime_rate >= 200:
        risk_factors.append("City-level women-crime rate is very high compared with other cities in the dataset.")
    elif crime_rate >= 100:
        risk_factors.append("City-level women-crime rate is elevated.")
    elif crime_rate < 50:
        mitigating_factors.append("City-level women-crime rate is relatively low in this dataset.")

    police_score = int(row.get("police_access_score", 0))
    nearest_police = float(row.get("nearest_police_station_km", 0))

    if police_score <= 1:
        risk_factors.append(f"Limited nearby police access in the selected context ({nearest_police:.2f} km).")
    elif police_score >= 2:
        mitigating_factors.append("Police access score is moderate or better.")

    transport_score = int(row.get("transport_access_score", 0))
    if transport_score <= 1:
        risk_factors.append("Limited public transport access may reduce safe mobility options.")
    elif transport_score >= 3:
        mitigating_factors.append("Public transport access is relatively strong.")

    growth = float(row.get("women_crime_growth_21_23", 0))
    if growth > 0.15:
        risk_factors.append("City-level women-crime records show an increasing trend.")
    elif growth < -0.10:
        mitigating_factors.append("City-level women-crime records show a decreasing trend.")

    lighting_available = bool(int(row.get("lighting_data_available", 0)))
    lighting_score = int(row.get("lighting_score", 0))

    if not lighting_available:
        limitations.append("Street-light data is sparse or unavailable in OSM, so lighting context may be incomplete.")
    else:
        if lighting_score <= 2:
            risk_factors.append("Lighting score is low in the selected context.")
        elif lighting_score >= 4:
            mitigating_factors.append("Lighting score is relatively strong in the selected context.")

    if not risk_factors:
        risk_factors.append("No dominant high-risk factor was found from available features.")

    if not mitigating_factors:
        mitigating_factors.append("No strong mitigating factor was found from available features.")

    if not limitations:
        limitations.append("This is based on public city-level and OSM-derived data, not real-time ground truth.")

    return risk_factors, mitigating_factors, limitations


# ============================================================
# App title
# ============================================================

st.title("🛡️ NariSafe Risk Awareness")
st.caption(
    "Public-data-based women safety risk-awareness prototype. "
    "This is not a real-time crime prediction or emergency advisory system."
)


# ============================================================
# Load resources
# ============================================================

with st.spinner("Loading model and dataset from Hugging Face..."):
    bundle = load_model_bundle()
    df = load_lookup_data()
    config = load_feature_config()


# ============================================================
# Prepare model objects
# ============================================================

pipeline = bundle["pipeline"]
selected_threshold = float(bundle["selected_threshold"])

target_col = config["target_column"]
categorical_features = config["categorical_features"]
numeric_features = config["numeric_features"]

# Important:
# This model was trained without city, latitude, longitude.
feature_cols = categorical_features + numeric_features


# ============================================================
# Sidebar with dependent dropdowns
# This prevents invalid combinations like hour 14 if not present.
# ============================================================

with st.sidebar:
    st.header("Input Context")

    cities = sorted(df["city"].dropna().unique().tolist())
    city = st.selectbox("City", cities)

    city_df = df[df["city"] == city]

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    available_days = sorted(city_df["day_of_week"].dropna().unique().tolist())
    days = [day for day in day_order if day in available_days]

    day_of_week = st.selectbox("Day of Week", days)

    day_df = city_df[city_df["day_of_week"] == day_of_week]

    area_options = sorted(day_df["area_context"].dropna().unique().tolist())

    area_context = st.selectbox(
        "Area Context",
        area_options,
        format_func=pretty_label,
    )

    area_df = day_df[day_df["area_context"] == area_context]

    complaint_options = sorted(
        [
            complaint
            for complaint in area_df["complaint_type_clean"].dropna().unique().tolist()
            if complaint not in PRIVATE_OR_NOT_DEMO_COMPLAINTS
        ]
    )

    complaint_type_clean = st.selectbox(
        "Complaint Type",
        complaint_options,
        format_func=pretty_label,
    )

    complaint_df = area_df[
        area_df["complaint_type_clean"] == complaint_type_clean
    ]

    hour_options = sorted(
        complaint_df["hour"].dropna().astype(int).unique().tolist()
    )

    hour = st.selectbox(
        "Hour",
        hour_options,
        format_func=lambda h: f"{h}:00",
    )

    predict_clicked = st.button("Predict Risk Awareness", use_container_width=True)


st.markdown("---")


# ============================================================
# Prediction
# ============================================================

if predict_clicked:
    match = df[
        (df["city"] == city)
        & (df["day_of_week"] == day_of_week)
        & (df["hour"].astype(int) == int(hour))
        & (df["area_context"] == area_context)
        & (df["complaint_type_clean"] == complaint_type_clean)
    ]

    if match.empty:
        st.error(
            "No matching row found for this input combination. "
            "Try another city, area, complaint type, or hour."
        )
        st.stop()

    row_df = match.iloc[[0]].copy()
    row = match.iloc[0]

    prediction, confidence = threshold_predict(
        pipeline=pipeline,
        selected_threshold=selected_threshold,
        feature_cols=feature_cols,
        row_df=row_df,
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Risk Awareness Level")
        show_risk_badge(prediction)
        st.metric("High-risk threshold", f"{selected_threshold:.2f}")

    with col2:
        st.subheader("Model Confidence")

        high_conf = confidence.get("high", 0.0)
        medium_conf = confidence.get("medium", 0.0)
        low_conf = confidence.get("low", 0.0)

        st.write(f"High: **{high_conf * 100:.2f}%**")
        st.progress(high_conf)

        st.write(f"Medium: **{medium_conf * 100:.2f}%**")
        st.progress(medium_conf)

        st.write(f"Low: **{low_conf * 100:.2f}%**")
        st.progress(low_conf)

    st.markdown("---")

    st.subheader("Context Breakdown")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("City", city)
        st.metric("Women Crime Rate", f"{float(row.get('women_crime_rate_2023', 0)):.1f}/lakh")
        st.metric("Women Crime Total", int(row.get("women_crime_2023", 0)))
        st.metric("Growth 2021-23", f"{float(row.get('women_crime_growth_21_23', 0)) * 100:.1f}%")

    with c2:
        st.metric("Day", day_of_week)
        st.metric("Hour", f"{int(hour)}:00")
        st.metric("Time Bucket", pretty_label(row.get("time_bucket", "")))
        st.metric("Weekend", "Yes" if int(row.get("is_weekend", 0)) == 1 else "No")

    with c3:
        st.metric("Area Context", pretty_label(area_context))
        st.metric("Crowd Density", pretty_label(row.get("crowd_density", "")))
        st.metric("Severity By Crime Type", f"{int(row.get('complaint_severity', 0))}/3")
        st.metric("Complaint Share", f"{float(row.get('complaint_type_share', 0)) * 100:.2f}%")

    with c4:
        st.metric("Lighting", lighting_display(row))
        st.metric("Police Score", f"{int(row.get('police_access_score', 0))}/3")
        st.metric("Nearest Police", f"{float(row.get('nearest_police_station_km', 0)):.2f} km")
        st.metric("Transport Score", f"{int(row.get('transport_access_score', 0))}/3")

    risk_factors, mitigating_factors, limitations = build_explanations(row)

    st.markdown("---")

    e1, e2, e3 = st.columns(3)

    with e1:
        st.subheader("Key Risk Factors")
        for item in risk_factors:
            st.write(f"- {item}")

    with e2:
        st.subheader("Mitigating Factors")
        for item in mitigating_factors:
            st.write(f"- {item}")

    with e3:
        st.subheader("Data Limitations")
        for item in limitations:
            st.write(f"- {item}")

else:
    st.info("Choose input values from the sidebar and click Predict Risk Awareness.")


# ============================================================
# Footer
# ============================================================

st.markdown("---")

st.caption(
    "Ethical use: This prototype is for educational/demo use only. "
    "Do not use it as an emergency decision system, policing tool, official safety advisory, "
    "individual risk scoring system, or real-time crime prediction system."
)
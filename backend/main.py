"""
NariSafe Risk Predictor — FastAPI Backend

Routes:
  GET  /           → serves frontend/index.html
  GET  /health     → model + dataset status
  GET  /metadata   → dropdown values for the UI
  POST /predict    → risk prediction for city/day/hour/area/complaint

The model is a threshold-tuned no-location sklearn Pipeline loaded from
models/no_location_baselines/best_no_location_threshold_model.joblib.
Prediction rows are looked up by (city, day_of_week, hour, area_context,
complaint_type_clean) from the pre-built feature CSV.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
CSV_PATH     = ROOT / "data/processed/ml_ready/narisafe_ml_features.csv"
MODEL_PATH   = ROOT / "models/no_location_baselines/best_no_location_threshold_model.joblib"
CONFIG_PATH  = ROOT / "data/processed/ml_ready_no_location/feature_config_no_location.json"
FRONTEND_DIR = ROOT / "frontend"

# ── App-level state (populated at startup) ────────────────────────────────────
_state: dict = {}

DAYS_ORDERED = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

DISCLAIMER = (
    "This is a public-data-based risk-awareness prototype using proxy labels, "
    "not guaranteed real-world crime prediction. Risk scores are derived from "
    "NCRB 2021-2023 statistics and OpenStreetMap infrastructure data using "
    "rule-based labels. Use this as an awareness tool only — not as a "
    "substitute for official safety advisories."
)
WARNING = (
    "Predictions are city-level aggregates and do not reflect hyper-local "
    "conditions. Actual risk at any specific location may differ significantly."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_label(snake: str) -> str:
    """snake_case → human-readable label."""
    label = " ".join(w.capitalize() for w in snake.split("_"))
    for old, new in [
        ("Ipc", "IPC"), ("Sec ", "Sec. "), ("Kna ", "K&A "),
        ("Itp ", "ITP "), ("Pocso", "POCSO"),
        ("&Amp;", "&"), ("18 Yrs", "18 Yrs"),
    ]:
        label = label.replace(old, new)
    return label


def _load_resources() -> None:
    bundle = joblib.load(MODEL_PATH)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    feature_cols = cfg["numeric_features"] + cfg["categorical_features"]

    df = pd.read_csv(CSV_PATH)
    df_idx = df.set_index(
        ["city", "day_of_week", "hour", "area_context", "complaint_type_clean"]
    ).sort_index()

    _state["pipeline"]           = bundle["pipeline"]
    _state["selected_threshold"] = bundle["selected_threshold"]
    _state["class_labels"]       = bundle["class_labels"]
    _state["feature_cols"]       = feature_cols
    _state["df"]                 = df_idx
    _state["metadata"]           = {
        "cities": sorted(df["city"].unique().tolist()),
        "days":   DAYS_ORDERED,
        "hours":  sorted(int(h) for h in df["hour"].unique()),
        "area_contexts": sorted(df["area_context"].unique().tolist()),
        "complaint_types": [
            {"value": ct, "label": _to_label(ct)}
            for ct in sorted(df["complaint_type_clean"].unique())
        ],
    }


def _build_explanation(row: dict) -> dict:
    """
    Return structured explanation split into three buckets:
      risk_factors      — things that increase risk
      mitigating_factors — things that reduce risk
      data_limitations  — missing or sparse data caveats
    """
    risk: list[str] = []
    mit:  list[str] = []
    lim:  list[str] = []

    # ── Time ─────────────────────────────────────────────────────────────────
    tb = row.get("time_bucket", "")
    if tb == "late_night":
        risk.append("Late-night hour — highest-risk time window")
    elif tb == "evening":
        risk.append("Evening hour — elevated-risk time window")
    elif tb in ("morning", "afternoon"):
        mit.append(f"Daytime hour ({tb}) — generally lower risk")

    # ── Complaint severity ────────────────────────────────────────────────────
    sev = int(row.get("complaint_severity", 1))
    ct  = row.get("complaint_type_clean", "").replace("_", " ")
    if sev == 3:
        risk.append(f"High-severity crime type: {ct}")
    elif sev == 2:
        risk.append(f"Medium-severity crime type: {ct}")
    else:
        mit.append(f"Lower-severity crime type: {ct}")

    # ── Complaint share ───────────────────────────────────────────────────────
    share = float(row.get("complaint_type_share", 0))
    if share >= 0.20:
        risk.append(
            f"Crime type represents {share*100:.1f}% of all city crimes — "
            "very high concentration"
        )
    elif share >= 0.05:
        risk.append(f"Crime type accounts for {share*100:.1f}% of city crimes")
    elif share > 0:
        mit.append(
            f"Crime type has a low share ({share*100:.2f}%) of city crimes"
        )
    else:
        lim.append(
            "No recorded cases of this crime type in the city (2023 NCRB data)"
        )

    # ── Crowd density ─────────────────────────────────────────────────────────
    crowd = row.get("crowd_density", "medium")
    area  = row.get("area_context", "")
    if crowd == "low":
        risk.append(
            f"Low crowd density in {area} area — reduced natural surveillance"
        )
    elif crowd == "high":
        mit.append(
            f"High crowd density in {area} area — increased natural surveillance"
        )

    # ── City crime rate ───────────────────────────────────────────────────────
    cr = float(row.get("women_crime_rate_2023", 0))
    if cr >= 200:
        risk.append(f"City has very high women's crime rate: {cr:.0f} per lakh population")
    elif cr >= 100:
        risk.append(f"City has elevated women's crime rate: {cr:.0f} per lakh population")
    elif cr < 50:
        # Low city crime rate is a mitigating factor, not a risk factor
        mit.append(f"City has relatively low women's crime rate: {cr:.0f} per lakh population")

    # ── Lighting ──────────────────────────────────────────────────────────────
    ls = int(row.get("lighting_score", 3))
    sl = int(row.get("street_light_count_2km", 0))
    if sl == 0:
        lim.append(
            "Street-light data is sparse or unavailable in OSM, "
            "so lighting context may be incomplete."
        )
    elif ls <= 2:
        risk.append(f"Low lighting score ({ls}/5) — poorly lit area increases risk")
    elif ls >= 4:
        mit.append(f"Good lighting score ({ls}/5) — well-lit area")

    # ── Police access ─────────────────────────────────────────────────────────
    ps   = int(row.get("police_access_score", 0))
    dist = float(row.get("nearest_police_station_km", 0))
    if ps == 0:
        risk.append(f"Poor police access — nearest station {dist:.1f} km away")
    elif ps == 1:
        risk.append(f"Limited police access — nearest station {dist:.1f} km")
    elif ps == 3:
        mit.append(f"Good police access — nearest station {dist:.2f} km away")
    else:
        mit.append(f"Moderate police access — nearest station {dist:.1f} km")

    # ── Crime growth ──────────────────────────────────────────────────────────
    growth = float(row.get("women_crime_growth_21_23", 0))
    if growth > 0.15:
        risk.append(f"City crime trending up: +{growth*100:.0f}% since 2021")
    elif growth < -0.10:
        mit.append(f"City crime declining: {growth*100:.0f}% since 2021")

    # ── Transport ─────────────────────────────────────────────────────────────
    ts = int(row.get("transport_access_score", 0))
    if ts == 0:
        risk.append("No public transport access — isolated area")
    elif ts >= 3:
        mit.append("Good public transport availability in the area")

    return {
        "key_risk_factors":   risk,
        "mitigating_factors": mit,
        "data_limitations":   lim,
    }


def _explanation_context(row: dict) -> dict:
    return {
        "city_context": {
            "women_crime_rate_2023":   float(row["women_crime_rate_2023"]),
            "women_crime_2023":        int(row["women_crime_2023"]),
            "women_crime_growth_pct":  round(float(row["women_crime_growth_21_23"]) * 100, 1),
            "population_lakhs":        float(row["population_lakhs"]),
            "chargesheeting_rate":     float(row["chargesheeting_rate_2023"]),
        },
        "time_context": {
            "day_of_week":  row["day_of_week"],
            "hour":         int(row["hour"]),
            "time_bucket":  row["time_bucket"],
            "is_weekend":   bool(int(row["is_weekend"])),
        },
        "area_context": {
            "area_type":     row["area_context"],
            "crowd_density": row["crowd_density"],
        },
        "complaint_context": {
            "complaint_type":         _to_label(row["complaint_type_clean"]),
            "severity_level":         int(row["complaint_severity"]),
            "cases_in_city_2023":     int(row["complaint_type_count"]),
            "share_of_city_crimes_pct": round(float(row["complaint_type_share"]) * 100, 2),
        },
        "infrastructure": {
            "lighting_score":          int(row["lighting_score"]),
            "lighting_data_available": bool(int(row.get("lighting_data_available", 0))),
            "police_access_score":     int(row["police_access_score"]),
            "nearest_police_km":       round(float(row["nearest_police_station_km"]), 2),
            "transport_access_score":  int(row["transport_access_score"]),
            "police_stations_5km":     int(row["police_station_count_5km"]),
        },
    }


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_resources()
    pipeline   = _state["pipeline"]
    model_name = type(pipeline.named_steps["model"]).__name__
    threshold  = _state["selected_threshold"]
    n_rows     = len(_state["df"])
    print(f"✓ Model loaded   : {model_name}")
    print(f"✓ Threshold      : {threshold}")
    print(f"✓ Dataset loaded : {n_rows:,} rows indexed")
    print(f"✓ Features       : {len(_state['feature_cols'])}")
    yield


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="NariSafe Risk Predictor API",
    description="Women safety risk prediction using NCRB data + OSM features",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    city: str
    day_of_week: str
    hour: int
    area_context: str
    complaint_type_clean: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_ui():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        return {"message": "Frontend not found. Run from project root."}
    return FileResponse(str(index))


@app.get("/health")
def health():
    pipeline = _state.get("pipeline")
    df       = _state.get("df")
    return {
        "status":             "ok",
        "model_type":         type(pipeline.named_steps["model"]).__name__ if pipeline else None,
        "model_classes":      pipeline.classes_.tolist() if pipeline else None,
        "selected_threshold": _state.get("selected_threshold"),
        "dataset_rows":       len(df) if df is not None else 0,
        "feature_count":      len(_state.get("feature_cols", [])),
    }


@app.get("/metadata")
def metadata():
    return _state["metadata"]


@app.post("/predict")
def predict(req: PredictRequest):
    key = (req.city, req.day_of_week, req.hour, req.area_context, req.complaint_type_clean)

    df: pd.DataFrame = _state["df"]
    try:
        row_series = df.loc[key]
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No matching row for: city={req.city!r}, "
                f"day={req.day_of_week!r}, hour={req.hour}, "
                f"area={req.area_context!r}, type={req.complaint_type_clean!r}"
            ),
        )

    # Reconstruct full row dict (index values were separated during set_index)
    row = row_series.to_dict()
    row.update({
        "city": req.city,
        "day_of_week": req.day_of_week,
        "hour": req.hour,
        "area_context": req.area_context,
        "complaint_type_clean": req.complaint_type_clean,
    })

    feature_cols: list[str] = _state["feature_cols"]
    X = pd.DataFrame([{col: row[col] for col in feature_cols}])

    pipeline           = _state["pipeline"]
    selected_threshold = _state["selected_threshold"]
    class_labels       = _state["class_labels"]

    proba   = pipeline.predict_proba(X)[0]
    classes = pipeline.classes_.tolist()
    raw_confidence = {cls: float(p) for cls, p in zip(classes, proba)}
    display_confidence = {cls: round(p, 4) for cls, p in raw_confidence.items()}

    high_prob = raw_confidence.get("high", 0.0)
    if high_prob >= selected_threshold:
        risk_awareness_level = "high"
    else:
        low_prob    = display_confidence.get("low", 0.0)
        medium_prob = display_confidence.get("medium", 0.0)
        risk_awareness_level = "low" if low_prob >= medium_prob else "medium"

    explanation = _build_explanation(row)
    context = _explanation_context(row)

    return {
        # Frontend response contract
        "predicted_risk_level": risk_awareness_level,
        "probabilities":          display_confidence,
        "explanation_context":    context,
        "explanation": {
            "risk_factors":       explanation["key_risk_factors"],
            "mitigating_factors": explanation["mitigating_factors"],
            "data_limitations":   explanation["data_limitations"],
        },
        "warning":                WARNING,
        "disclaimer":             DISCLAIMER,

        # Backward-compatible API fields used by earlier scripts/clients
        "risk_awareness_level": risk_awareness_level,
        "model_confidence":     display_confidence,
        "selected_threshold":   selected_threshold,
        "context":              context,
        "key_risk_factors":     explanation["key_risk_factors"],
        "mitigating_factors":   explanation["mitigating_factors"],
        "data_limitations":     explanation["data_limitations"],
    }

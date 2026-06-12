"""
app.py — NariSafe Streamlit frontend
Run : streamlit run app.py
Loads model + dataset directly from Hugging Face — no separate backend needed.
"""

import json

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="NariSafe · Women Safety Risk Awareness",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Constants ──────────────────────────────────────────────────────────────────
HF_MODEL_REPO   = "avnisinghal001/narisafe-risk-awareness-model"
HF_DATASET_REPO = "avnisinghal001/narisafe-risk-awareness-dataset"

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

HOUR_LABELS = {
    6:  "6:00 AM — Early Morning",
    9:  "9:00 AM — Morning",
    12: "12:00 PM — Noon",
    15: "3:00 PM — Afternoon",
    18: "6:00 PM — Evening",
    21: "9:00 PM — Night",
    23: "11:00 PM — Late Night",
}

DOMESTIC_TYPES = {
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

DISCLAIMER = (
    "This is a public-data-based risk-awareness prototype using proxy labels, "
    "not guaranteed real-world crime prediction. Risk scores are derived from "
    "NCRB 2021–2023 statistics and OpenStreetMap infrastructure data using "
    "rule-based labels. Use this as an awareness tool only — not as a substitute "
    "for official safety advisories."
)


# ── Custom CSS ─────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Streamlit chrome ─────────────────────────────────────────────────────── */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
    visibility: hidden !important;
    height: 0 !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Root & body ──────────────────────────────────────────────────────────── */
html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #FDF4F9 !important;
}
.block-container {
    padding-top: 0 !important;
    padding-bottom: 60px !important;
    max-width: 860px !important;
}

/* ── Select labels ────────────────────────────────────────────────────────── */
.stSelectbox > label, .stSelectbox label p {
    font-size: 0.73rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
    color: #9D174D !important;
    margin-bottom: 2px !important;
}

/* ── Select boxes ─────────────────────────────────────────────────────────── */
.stSelectbox [data-baseweb="select"] > div:first-child {
    border: 1.5px solid #FECDD3 !important;
    border-radius: 6px !important;
    background: #FFFFFF !important;
    font-size: 0.9rem !important;
    color: #1C1917 !important;
    min-height: 44px !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
.stSelectbox [data-baseweb="select"] > div:first-child:focus-within {
    border-color: #BE185D !important;
    box-shadow: 0 0 0 3px rgba(190,24,93,0.12) !important;
}

/* ── Button ───────────────────────────────────────────────────────────────── */
div[data-testid="stButton"] > button {
    width: 100% !important;
    background: #9D174D !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 14px 28px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 6px 24px rgba(157, 23, 77, 0.38) !important;
    transition: opacity 0.15s, transform 0.12s, box-shadow 0.15s !important;
    margin-top: 8px !important;
}
div[data-testid="stButton"] > button:hover {
    background: #BE185D !important;
    opacity: 1 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(157, 23, 77, 0.45) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Spinner ──────────────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #BE185D !important; }

/* ── Error / info alerts ──────────────────────────────────────────────────── */
.stAlert { border-radius: 6px !important; font-size: 0.88rem !important; }

/* ── Divider ──────────────────────────────────────────────────────────────── */
hr { border-color: #FCE7F3 !important; }
</style>
""", unsafe_allow_html=True)


# ── HF resource loading ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_bundle():
    path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename="best_no_location_threshold_model.joblib",
        repo_type="model",
    )
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def _load_df():
    path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename="narisafe_ml_features.csv",
        repo_type="dataset",
    )
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _load_config():
    path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename="feature_config_no_location.json",
        repo_type="model",
    )
    with open(path) as f:
        return json.load(f)


# ── ML helpers ─────────────────────────────────────────────────────────────────
def _predict(pipeline, threshold, feature_cols, row_df):
    X = row_df[feature_cols]
    probs   = pipeline.predict_proba(X)[0]
    classes = list(pipeline.classes_)
    conf    = {c: float(p) for c, p in zip(classes, probs)}
    high_p  = conf.get("high", 0.0)
    if high_p >= threshold:
        label = "high"
    else:
        label = "low" if conf.get("low", 0.0) >= conf.get("medium", 0.0) else "medium"
    return label, conf


def _snake_to_title(s: str) -> str:
    label = " ".join(w.capitalize() for w in str(s).split("_"))
    for old, new in [
        ("Ipc", "IPC"), ("Sec ", "Sec. "), ("Kna ", "K&A "),
        ("Itp ", "ITP "), ("Pocso", "POCSO"), ("&Amp;", "&"),
    ]:
        label = label.replace(old, new)
    return label


# ── HTML helpers ───────────────────────────────────────────────────────────────
def _sev_badge(n: int) -> str:
    if n >= 3:
        return ('<span style="background:#FEE2E2;color:#B91C1C;padding:2px 11px;'
                'border-radius:5px;font-size:.75rem;font-weight:700;">High</span>')
    if n == 2:
        return ('<span style="background:#FEF3C7;color:#92400E;padding:2px 11px;'
                'border-radius:5px;font-size:.75rem;font-weight:700;">Medium</span>')
    return ('<span style="background:#D1FAE5;color:#065F46;padding:2px 11px;'
            'border-radius:5px;font-size:.75rem;font-weight:700;">Low</span>')


def _dots(val: int, max_val: int, color: str = "#BE185D") -> str:
    return "".join(
        f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
        f'background:{color if i < val else "#E5E7EB"};margin-right:3px;"></span>'
        for i in range(max_val)
    )


def _ctx_card(title: str, rows: list) -> str:
    rows_html = "".join(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:7px 0;border-bottom:1px solid #FDF2F8;font-size:0.83rem;">'
        f'<span style="color:#9CA3AF;flex-shrink:0;">{k}</span>'
        f'<span style="font-weight:600;color:#1C1917;text-align:right;max-width:58%;'
        f'word-break:break-word;">{v}</span></div>'
        for k, v in rows
    )
    return (
        f'<div style="background:#FFFFFF;border:1px solid #FCE7F3;border-radius:8px;'
        f'padding:18px 20px;height:100%;box-shadow:0 2px 16px rgba(190,24,93,0.07);">'
        f'<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.8px;color:#BE185D;margin-bottom:14px;">{title}</div>'
        f'{rows_html}</div>'
    )


_RISK_ICONS = [
    (["late-night", "evening"],              "🌙"),
    (["severity"],                           "⚠️"),
    (["concentration", "share", "%"],        "📊"),
    (["crowd"],                              "👥"),
    (["crime rate", "very high", "elevated"],"🏙️"),
    (["lighting", "poorly lit"],             "💡"),
    (["police", "station"],                  "🚔"),
    (["transport", "isolated"],              "🚌"),
    (["trending", "increasing"],             "📈"),
]
_MIT_ICONS = [
    (["daytime"],                            "☀️"),
    (["lower"],                              "✅"),
    (["low share", "low.*crime", "low.*rate","relatively low"], "📉"),
    (["crowd"],                              "👥"),
    (["lighting", "well-lit"],               "💡"),
    (["police", "station"],                  "🚔"),
    (["transport"],                          "🚌"),
    (["declining", "decreasing"],            "📉"),
]

def _icon(text: str, icon_map: list) -> str:
    t = text.lower()
    for keywords, icon in icon_map:
        if any(k in t for k in keywords):
            return icon
    return "•"


def _expl_item(text: str, kind: str) -> str:
    styles = {
        "risk": ("#FEF2F2", "#7F1D1D"),
        "mit":  ("#ECFDF5", "#14532D"),
        "lim":  ("#EFF6FF", "#1E3A8A"),
    }
    bg, fg = styles.get(kind, ("#F3F4F6", "#374151"))
    icon = (_icon(text, _RISK_ICONS) if kind == "risk"
            else _icon(text, _MIT_ICONS) if kind == "mit"
            else "ℹ️")
    return (
        f'<li style="display:flex;align-items:flex-start;gap:10px;background:{bg};'
        f'color:{fg};border-radius:6px;padding:10px 14px;font-size:0.85rem;'
        f'line-height:1.55;list-style:none;margin-bottom:7px;">'
        f'<span style="flex-shrink:0;font-size:0.95rem;margin-top:1px;">{icon}</span>'
        f'{text}</li>'
    )


# ── Explanation builder ────────────────────────────────────────────────────────
def _build_explanation(row) -> tuple[list, list, list]:
    risk, mit, lim = [], [], []

    tb = str(row.get("time_bucket", ""))
    if tb == "late_night":
        risk.append("Late-night hour — highest-risk time window")
    elif tb == "evening":
        risk.append("Evening hour — elevated-risk time window")
    elif tb in ("morning", "afternoon"):
        mit.append(f"Daytime hour ({tb}) — generally lower risk")

    sev = int(row.get("complaint_severity", 1))
    ct  = _snake_to_title(row.get("complaint_type_clean", ""))
    if sev == 3:
        risk.append(f"High-severity crime type: {ct}")
    elif sev == 2:
        risk.append(f"Medium-severity crime type: {ct}")
    else:
        mit.append(f"Lower-severity crime type: {ct}")

    share = float(row.get("complaint_type_share", 0))
    if share >= 0.20:
        risk.append(f"Crime type represents {share*100:.1f}% of all city crimes — very high concentration")
    elif share >= 0.05:
        risk.append(f"Crime type accounts for {share*100:.1f}% of city crimes")
    elif share > 0:
        mit.append(f"Crime type has low share ({share*100:.2f}%) of city crimes")
    else:
        lim.append("No recorded cases of this crime type in the city (2023 NCRB data)")

    crowd = str(row.get("crowd_density", ""))
    area  = str(row.get("area_context", ""))
    if crowd == "low":
        risk.append(f"Low crowd density in {area} area — reduced natural surveillance")
    elif crowd == "high":
        mit.append(f"High crowd density in {area} area — increased natural surveillance")

    cr = float(row.get("women_crime_rate_2023", 0))
    if cr >= 200:
        risk.append(f"City has very high women's crime rate: {cr:.0f} per lakh population")
    elif cr >= 100:
        risk.append(f"City has elevated women's crime rate: {cr:.0f} per lakh population")
    elif cr < 50:
        mit.append(f"City has relatively low women's crime rate: {cr:.0f} per lakh population")

    ls = int(row.get("lighting_score", 3))
    sl = int(row.get("street_light_count_2km", 0))
    if sl == 0:
        lim.append("Street-light data is sparse or unavailable in OSM, so lighting context may be incomplete.")
    elif ls <= 2:
        risk.append(f"Low lighting score ({ls}/5) — poorly lit area increases risk")
    elif ls >= 4:
        mit.append(f"Good lighting score ({ls}/5) — well-lit area")

    ps   = int(row.get("police_access_score", 0))
    dist = float(row.get("nearest_police_station_km", 0))
    if ps == 0:
        risk.append(f"Poor police access — nearest station {dist:.1f} km away")
    elif ps == 1:
        risk.append(f"Limited police access — nearest station {dist:.1f} km")
    elif ps >= 3:
        mit.append(f"Good police access — nearest station {dist:.2f} km away")
    else:
        mit.append(f"Moderate police access — nearest station {dist:.1f} km")

    growth = float(row.get("women_crime_growth_21_23", 0))
    if growth > 0.15:
        risk.append(f"City crime trending up: +{growth*100:.0f}% since 2021")
    elif growth < -0.10:
        mit.append(f"City crime declining: {growth*100:.0f}% since 2021")

    ts = int(row.get("transport_access_score", 0))
    if ts == 0:
        risk.append("No public transport access — isolated area")
    elif ts >= 3:
        mit.append("Good public transport availability in the area")

    return risk, mit, lim


# ── Result renderer ────────────────────────────────────────────────────────────
def _render_result(label: str, conf: dict, threshold: float, row) -> None:
    _BADGE = {
        "high":   ("#FEF2F2", "#DC2626", "#DC2626", "🔴", "HIGH RISK",
                   "Elevated contextual risk — exercise extra caution"),
        "medium": ("#FFFBEB", "#D97706", "#D97706", "🟡", "MEDIUM RISK",
                   "Moderate contextual risk — stay aware of your surroundings"),
        "low":    ("#ECFDF5", "#059669", "#059669", "🟢", "LOW RISK",
                   "Lower contextual risk — conditions appear relatively safer"),
    }
    bg, fg, border, icon, rlabel, rdesc = _BADGE.get(
        label, ("#F3F4F6", "#374151", "#374151", "⚪", label.upper(), "")
    )

    # ── Confidence bars ────────────────────────────────────────────────────────
    _BAR_CLR = {"high": "#DC2626", "medium": "#D97706", "low": "#059669"}
    bars = ""
    for cls in ["high", "medium", "low"]:
        pct = conf.get(cls, 0) * 100
        c   = _BAR_CLR[cls]
        bars += (
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:11px;">'
            f'<span style="font-size:0.81rem;font-weight:700;text-transform:capitalize;'
            f'width:64px;flex-shrink:0;color:{c};">{cls.capitalize()}</span>'
            f'<div style="flex:1;background:#F3F4F6;border-radius:4px;height:11px;overflow:hidden;">'
            f'<div style="width:{pct:.1f}%;background:{c};height:100%;border-radius:4px;"></div>'
            f'</div>'
            f'<span style="font-size:0.81rem;font-weight:700;width:48px;text-align:right;'
            f'color:{c};flex-shrink:0;">{pct:.1f}%</span></div>'
        )

    st.markdown(f"""
<div style="background:#FFFFFF;border-radius:8px;
            padding:32px 32px 28px;margin:28px 0 0;border:1px solid #FCE7F3;
            box-shadow:0 6px 32px rgba(190,24,93,0.09);">

  <div style="text-align:center;margin-bottom:28px;">
    <div style="font-size:0.7rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;
                color:#C084AC;margin-bottom:16px;">Risk Awareness Level</div>
    <div style="display:inline-flex;align-items:center;gap:14px;padding:16px 48px;
                border-radius:8px;background:{bg};border:2px solid {border};">
      <span style="font-size:1.8rem;">{icon}</span>
      <span style="font-size:1.5rem;font-weight:800;letter-spacing:1px;color:{fg};">{rlabel}</span>
    </div>
    <div style="font-size:0.87rem;color:#78716C;margin-top:12px;">{rdesc}</div>
    <div style="font-size:0.75rem;color:#C084AC;margin-top:4px;">
      High-risk threshold: P(high) ≥ {threshold} &nbsp;·&nbsp; threshold-tuned no-location model
    </div>
  </div>

  <hr style="border:none;border-top:1px solid #FCE7F3;margin:0 0 22px;" />

  <div style="font-size:0.88rem;font-weight:700;color:#9D174D;margin-bottom:14px;
              letter-spacing:0.3px;">Model Confidence</div>
  {bars}
</div>
""", unsafe_allow_html=True)

    # ── Context grid ───────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:1rem;font-weight:700;color:#1C1917;margin:26px 0 14px;">'
        '📋 Context Breakdown</div>',
        unsafe_allow_html=True,
    )

    growth_pct = float(row.get("women_crime_growth_21_23", 0)) * 100
    growth_str = ("+" if growth_pct > 0 else "") + f"{growth_pct:.1f}%"

    lighting_available = bool(int(row.get("lighting_data_available", 0)))
    lighting_val = (
        f'{_dots(int(row.get("lighting_score", 0)), 5)} '
        f'({int(row.get("lighting_score", 0))}/5)'
        if lighting_available
        else "Unknown / estimated (OSM data unavailable)"
    )

    tb_raw  = str(row.get("time_bucket", "")).replace("_", " ").title()
    weekend = " 🗓️" if int(row.get("is_weekend", 0)) else ""

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown(_ctx_card("🏙️ City Crime (2023)", [
            ("Crime Rate",    f'{float(row.get("women_crime_rate_2023", 0)):.1f} / lakh'),
            ("Total Cases",   f'{int(row.get("women_crime_2023", 0)):,}'),
            ("Growth 21→23",  growth_str),
            ("Population",    f'{float(row.get("population_lakhs", 0))} Lakhs'),
            ("Chargesheeting",f'{float(row.get("chargesheeting_rate_2023", 0))}%'),
        ]), unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown(_ctx_card("📂 Complaint Type", [
            ("Type",        _snake_to_title(row.get("complaint_type_clean", ""))),
            ("Severity",    _sev_badge(int(row.get("complaint_severity", 1)))),
            ("Cases (2023)",f'{int(row.get("complaint_type_count", 0)):,}'),
            ("City Share",  f'{float(row.get("complaint_type_share", 0))*100:.2f}%'),
        ]), unsafe_allow_html=True)

    with col2:
        st.markdown(_ctx_card("⏰ Time & Area", [
            ("Day",         str(row.get("day_of_week", "")) + weekend),
            ("Hour",        f'{int(row.get("hour", 0))}:00'),
            ("Time Window", tb_raw),
            ("Area Type",   str(row.get("area_context", "")).title()),
            ("Crowd Density",str(row.get("crowd_density", "")).title()),
        ]), unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown(_ctx_card("🏗️ Infrastructure", [
            ("Lighting",      lighting_val),
            ("Police Access", f'{_dots(int(row.get("police_access_score", 0)), 3)} '
                              f'({int(row.get("police_access_score", 0))}/3)'),
            ("Nearest Station",f'{float(row.get("nearest_police_station_km", 0)):.2f} km'),
            ("Transport",     f'{_dots(int(row.get("transport_access_score", 0)), 3)} '
                              f'({int(row.get("transport_access_score", 0))}/3)'),
            ("Stations 5km",  str(int(row.get("police_station_count_5km", 0)))),
        ]), unsafe_allow_html=True)

    # ── Analysis ───────────────────────────────────────────────────────────────
    risk_f, mit_f, lim_f = _build_explanation(row)

    def _section(items, kind, heading_color, heading_icon, heading_text):
        if not items:
            return ""
        items_html = "".join(_expl_item(t, kind) for t in items)
        return (
            f'<div style="margin-bottom:20px;">'
            f'<div style="font-size:0.73rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.8px;color:{heading_color};margin-bottom:10px;">'
            f'{heading_icon} {heading_text}</div>'
            f'<ul style="margin:0;padding:0;">{items_html}</ul></div>'
        )

    analysis_html = (
        _section(risk_f, "risk", "#B91C1C", "🔺", "Key Risk Factors")
        + ('<hr style="border:none;border-top:1px dashed #FCE7F3;margin:0 0 18px;">'
           if risk_f and mit_f else "")
        + _section(mit_f, "mit", "#065F46", "✅", "Mitigating Factors")
        + ('<hr style="border:none;border-top:1px dashed #FCE7F3;margin:0 0 18px;">'
           if (risk_f or mit_f) and lim_f else "")
        + _section(lim_f, "lim", "#1E40AF", "ℹ️", "Data Limitations")
    )

    st.markdown(
        '<div style="font-size:1rem;font-weight:700;color:#1C1917;margin:26px 0 14px;">'
        '💡 Analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="background:#FFFFFF;border:1px solid #FCE7F3;border-radius:8px;'
        f'padding:24px 26px;box-shadow:0 2px 16px rgba(190,24,93,0.07);">'
        f'{analysis_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:#FFF7F0;border:1.5px solid #FDE68A;border-radius:8px;'
        f'padding:16px 20px;margin-top:20px;font-size:0.82rem;color:#78350F;line-height:1.65;">'
        f'<div style="font-weight:700;margin-bottom:5px;">⚠️ Disclaimer</div>'
        f'{DISCLAIMER}</div>'
        f'<div style="height:52px"></div>',
        unsafe_allow_html=True,
    )


# ── Main ────────────────────────────────────────────────────────────────────────
def main() -> None:
    _inject_css()

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown("""
<div style="background:#9D174D;
            padding:48px 28px 44px;text-align:center;margin:-4px -4px 0;
            border-radius:0 0 8px 8px;
            box-shadow:0 10px 48px rgba(157,23,77,0.32);">
  <div style="font-size:3.2rem;margin-bottom:10px;filter:drop-shadow(0 2px 8px rgba(0,0,0,.25));">🛡️</div>
  <h1 style="color:white;font-size:2.2rem;font-weight:800;letter-spacing:-0.6px;
             margin:0 0 10px;text-shadow:0 2px 12px rgba(0,0,0,.2);">NariSafe</h1>
  <p style="color:rgba(255,255,255,0.82);font-size:0.93rem;max-width:500px;
            margin:0 auto;line-height:1.65;">
    Women safety risk-awareness using NCRB 2021–2023 crime statistics
    &amp; OpenStreetMap urban infrastructure
  </p>
  <div style="display:flex;justify-content:center;gap:18px;margin-top:20px;flex-wrap:wrap;">
    <span style="background:rgba(255,255,255,0.15);color:rgba(255,255,255,0.9);
                 padding:5px 14px;border-radius:6px;font-size:0.76rem;font-weight:600;">
      34 Indian Cities
    </span>
    <span style="background:rgba(255,255,255,0.15);color:rgba(255,255,255,0.9);
                 padding:5px 14px;border-radius:6px;font-size:0.76rem;font-weight:600;">
      489,804 Scenarios
    </span>
    <span style="background:rgba(255,255,255,0.15);color:rgba(255,255,255,0.9);
                 padding:5px 14px;border-radius:6px;font-size:0.76rem;font-weight:600;">
      HistGradientBoosting · Threshold 0.2
    </span>
  </div>
</div>
<div style="height:32px"></div>
""", unsafe_allow_html=True)

    # ── Load resources ──────────────────────────────────────────────────────────
    with st.spinner("Loading model and dataset from Hugging Face…"):
        try:
            bundle = _load_bundle()
            df     = _load_df()
            cfg    = _load_config()
        except Exception as e:
            st.error(f"Could not load resources from Hugging Face: {e}")
            st.stop()

    pipeline  = bundle["pipeline"]
    threshold = float(bundle["selected_threshold"])
    feat_cols = cfg["categorical_features"] + cfg["numeric_features"]

    # ── Input form ──────────────────────────────────────────────────────────────
    st.markdown("""
<div style="font-size:1rem;font-weight:700;color:#1C1917;margin-bottom:6px;">
  🔍 Enter Context Details
</div>
<div style="font-size:0.82rem;color:#9CA3AF;margin-bottom:18px;">
  Dropdowns are linked — each selection filters the valid options below it.
</div>
""", unsafe_allow_html=True)

    # Row 1 — City | Day
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        cities = sorted(df["city"].dropna().unique().tolist())
        city   = st.selectbox("City", cities)
    city_df = df[df["city"] == city]
    with c2:
        avail_days = [d for d in DAY_ORDER if d in city_df["day_of_week"].unique()]
        day = st.selectbox("Day of Week", avail_days)
    day_df = city_df[city_df["day_of_week"] == day]

    # Row 2 — Area | Hour  (Hour depends on area + complaint, so deferred)
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        area_opts  = sorted(day_df["area_context"].dropna().unique().tolist())
        area_raw   = st.selectbox(
            "Area Context",
            area_opts,
            format_func=lambda x: x.replace("_", " ").title(),
        )
    area_df = day_df[day_df["area_context"] == area_raw]

    # Row 3 — Complaint Type (full width)
    complaint_opts_raw = sorted(
        c for c in area_df["complaint_type_clean"].dropna().unique()
        if c not in DOMESTIC_TYPES
    )
    complaint_raw = st.selectbox(
        "Complaint Type",
        complaint_opts_raw,
        format_func=_snake_to_title,
    )
    complaint_df = area_df[area_df["complaint_type_clean"] == complaint_raw]

    # Row 4 — Hour (depends on all above)
    with c4:
        hour_opts = sorted(complaint_df["hour"].dropna().astype(int).unique().tolist())
        hour = st.selectbox(
            "Hour of Day",
            hour_opts,
            format_func=lambda h: HOUR_LABELS.get(h, f"{h}:00"),
        )

    # ── Analyse button ──────────────────────────────────────────────────────────
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    go = st.button("🔍  Analyse Risk Awareness")

    # ── Prediction ──────────────────────────────────────────────────────────────
    if go:
        match = df[
            (df["city"] == city)
            & (df["day_of_week"] == day)
            & (df["hour"].astype(int) == int(hour))
            & (df["area_context"] == area_raw)
            & (df["complaint_type_clean"] == complaint_raw)
        ]
        if match.empty:
            st.error(
                "No matching row found for this combination. "
                "Try a different city, day, area, complaint type, or hour."
            )
            st.stop()

        row_df = match.iloc[[0]].copy()
        row    = match.iloc[0]

        with st.spinner("Running prediction…"):
            label, conf = _predict(pipeline, threshold, feat_cols, row_df)

        st.session_state["last_result"] = (label, conf, threshold, row)

    if "last_result" in st.session_state:
        label, conf, threshold, row = st.session_state["last_result"]
        _render_result(label, conf, threshold, row)
    else:
        st.markdown("""
<div style="background:#FFFFFF;border:1.5px dashed #FCE7F3;border-radius:8px;
            padding:40px 28px;text-align:center;margin-top:28px;color:#C084AC;">
  <div style="font-size:2.4rem;margin-bottom:12px;">🔍</div>
  <div style="font-size:1rem;font-weight:600;color:#9D174D;margin-bottom:6px;">
    Select a context above and click Analyse
  </div>
  <div style="font-size:0.84rem;color:#9CA3AF;">
    Results will appear here — risk level, model confidence, and full context breakdown.
  </div>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

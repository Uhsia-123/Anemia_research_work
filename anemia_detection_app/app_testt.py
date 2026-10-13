"""
AnemiaSense — Nutrition & Symptom Assessment
Research prototype (Streamlit). Not a medical diagnosis.

Run:  streamlit run app.py
Files required in this folder:
    anemia_mlp.keras, scaler.joblib, feature_columns.json
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# --------------------------------------------------------------------------
# 0. PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AnemiaSense — Nutrition & Symptom Assessment",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

# The exact 17-feature schema and order used when the model was trained.
EXPECTED_FEATURES = [
    "age",
    "gender",
    "bmi",
    "diet_omnivore",
    "diet_pescatarian",
    "diet_vegan",
    "diet_vegetarian",
    "iron_percent_rda",
    "vitamin_b12_percent_rda",
    "folate_percent_rda",
    "symptoms_count",
    "has_fatigue",
    "has_muscle_weakness",
    "has_numbness_tingling",
    "has_memory_problems",
    "has_pale_skin",
    "has_bone_pain",
]

# Same encoding as the existing working app.py (Female = 1, Male = 0).
GENDER_ENCODING = {"Female": 1, "Male": 0}
DIET_OPTIONS = ["Omnivore", "Vegetarian", "Vegan", "Pescatarian"]

# 0.5 is the cut-off used in the training notebook to report precision/recall.
# It is NOT clinically validated. Scores within BORDERLINE_BAND of it are shown
# as "borderline" (a presentation choice made by this app, not a validated rule).
DECISION_THRESHOLD = 0.5
BORDERLINE_BAND = 0.10

# Pages
P_DASH = "🏠  Dashboard"
P_PROFILE = "👤  Personal Profile"
P_FOOD = "🥗  Food & Nutrition"
P_SYMP = "🩺  Symptom Assessment"
P_RESULTS = "📊  Screening Results"
P_RECS = "🍛  Dietary Recommendations"
P_ABOUT = "🧪  About the Model"
PAGES = [P_DASH, P_PROFILE, P_FOOD, P_SYMP, P_RESULTS, P_RECS, P_ABOUT]

NUTRIENTS = [
    ("iron", "Iron", "mg"),
    ("b12", "Vitamin B12", "µg"),
    ("folate", "Folate", "µg DFE"),
]

# --------------------------------------------------------------------------
# 1. FOOD-COMPOSITION DATA
# --------------------------------------------------------------------------
# Nutrient values are per 100 g of the food in the stated form, rounded from
# USDA FoodData Central (SR Legacy). Please verify against the source before
# any use beyond this prototype. Folate is in µg Dietary Folate Equivalents
# (DFE). Indian preparations, local varieties and cooking methods can differ
# (see the Indian Food Composition Tables, IFCT 2017, for local data).
FOOD_SOURCE = (
    "USDA FoodData Central, SR Legacy (values per 100 g, rounded). "
    "https://fdc.nal.usda.gov"
)


@dataclass(frozen=True)
class Food:
    id: str
    category: str
    name: str
    measure: str
    grams: int  # default portion in grams (editable by the user)
    iron: float  # mg per 100 g
    b12: float  # µg per 100 g
    folate: float  # µg DFE per 100 g


CAT_CEREAL = "Cereals & staples"
CAT_PULSES = "Pulses, beans & lentils"
CAT_GREENS = "Leafy greens"
CAT_FRUIT = "Fruits"
CAT_EGGS = "Eggs"
CAT_DAIRY = "Dairy"
CAT_MEAT = "Meat"
CAT_FISH = "Fish & seafood"
CAT_FORT = "Fortified foods"

CATEGORIES = [
    CAT_CEREAL, CAT_PULSES, CAT_GREENS, CAT_FRUIT, CAT_EGGS,
    CAT_DAIRY, CAT_MEAT, CAT_FISH, CAT_FORT,
]

FOODS: list[Food] = [
    # Cereals and staples
    Food("atta", CAT_CEREAL, "Chapati / roti (whole-wheat flour, dry weight)",
         "1 roti ≈ 30 g dry atta", 30, 3.6, 0.0, 44),
    Food("rice", CAT_CEREAL, "White rice, cooked (unenriched)",
         "1 katori ≈ 150 g cooked", 150, 0.2, 0.0, 3),
    # Pulses (boiled)
    Food("lentil", CAT_PULSES, "Lentils / dal, boiled",
         "1 katori ≈ 150 g cooked", 150, 3.3, 0.0, 181),
    Food("chickpea", CAT_PULSES, "Chickpeas (chana), boiled",
         "1 katori ≈ 150 g cooked", 150, 2.9, 0.0, 172),
    Food("rajma", CAT_PULSES, "Kidney beans (rajma), boiled",
         "1 katori ≈ 150 g cooked", 150, 2.2, 0.0, 130),
    Food("moong", CAT_PULSES, "Mung beans (moong), boiled",
         "1 katori ≈ 150 g cooked", 150, 1.4, 0.0, 159),
    # Leafy greens (boiled)
    Food("spinach", CAT_GREENS, "Spinach (palak), boiled",
         "1 katori ≈ 100 g cooked", 100, 3.6, 0.0, 146),
    Food("amaranth", CAT_GREENS, "Amaranth leaves (chaulai), boiled",
         "1 katori ≈ 100 g cooked", 100, 2.3, 0.0, 85),
    # Fruits (raw)
    Food("banana", CAT_FRUIT, "Banana", "1 medium ≈ 118 g", 118, 0.3, 0.0, 20),
    Food("orange", CAT_FRUIT, "Orange", "1 medium ≈ 130 g", 130, 0.1, 0.0, 30),
    Food("apple", CAT_FRUIT, "Apple, with skin", "1 medium ≈ 180 g", 180, 0.1, 0.0, 3),
    Food("pomegranate", CAT_FRUIT, "Pomegranate seeds",
         "½ cup ≈ 87 g", 87, 0.3, 0.0, 38),
    Food("papaya", CAT_FRUIT, "Papaya", "1 katori cubes ≈ 150 g", 150, 0.3, 0.0, 37),
    Food("guava", CAT_FRUIT, "Guava", "1 medium ≈ 100 g", 100, 0.3, 0.0, 49),
    # Eggs
    Food("egg", CAT_EGGS, "Egg, hard-boiled", "1 large egg ≈ 50 g", 50, 1.2, 1.11, 44),
    # Dairy
    Food("milk", CAT_DAIRY, "Milk, whole", "1 glass (200 ml) ≈ 206 g", 206, 0.03, 0.45, 5),
    Food("curd", CAT_DAIRY, "Yogurt / curd, plain whole-milk",
         "1 katori ≈ 150 g", 150, 0.05, 0.37, 7),
    # Meat (cooked)
    Food("chicken", CAT_MEAT, "Chicken breast, roasted, skinless",
         "Palm-sized portion ≈ 100 g cooked", 100, 1.0, 0.34, 4),
    Food("goat", CAT_MEAT, "Goat / mutton, roasted",
         "Palm-sized portion ≈ 100 g cooked", 100, 3.7, 1.13, 5),
    # Fish (cooked / canned)
    Food("salmon", CAT_FISH, "Salmon, farmed Atlantic, cooked",
         "1 fillet ≈ 100 g", 100, 0.3, 3.05, 29),
    Food("sardine", CAT_FISH, "Sardines, canned in oil, drained",
         "½ small tin ≈ 85 g", 85, 2.9, 8.94, 12),
]

FOODS_BY_CAT = {c: [f for f in FOODS if f.category == c] for c in CATEGORIES}
FORT_SLOTS = 3

# --------------------------------------------------------------------------
# 2. REFERENCE DIETARY ALLOWANCES (US NIH / National Academies RDA)
# --------------------------------------------------------------------------
RDA_SOURCE = (
    "US NIH Office of Dietary Supplements / National Academies RDAs "
    "(non-pregnant, non-lactating)."
)


def reference_rda(age: int, sex: str) -> dict[str, float]:
    """RDA for iron (mg), vitamin B12 (µg) and folate (µg DFE)."""
    male = sex == "Male"
    if age <= 3:
        iron, b12, folate = 7, 0.9, 150
    elif age <= 8:
        iron, b12, folate = 10, 1.2, 200
    elif age <= 13:
        iron, b12, folate = 8, 1.8, 300
    elif age <= 18:
        iron, b12, folate = (11 if male else 15), 2.4, 400
    elif age <= 50:
        iron, b12, folate = (8 if male else 18), 2.4, 400
    else:
        iron, b12, folate = 8, 2.4, 400
    return {"iron": float(iron), "b12": float(b12), "folate": float(folate)}


# --------------------------------------------------------------------------
# 3. SYMPTOMS
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Symptom:
    key: str  # equals the model feature name
    label: str
    hint: str


SYMPTOMS = [
    Symptom("has_fatigue", "Fatigue or unusual tiredness", "Tired even after enough rest"),
    Symptom("has_muscle_weakness", "Muscle weakness", "Difficulty with usual physical tasks"),
    Symptom("has_numbness_tingling", "Numbness or tingling", "Often in hands or feet"),
    Symptom("has_memory_problems", "Memory or concentration problems", "Trouble focusing or remembering"),
    Symptom("has_pale_skin", "Pale skin", "Paler than usual"),
    Symptom("has_bone_pain", "Bone pain", "Persistent aching in bones"),
]

# --------------------------------------------------------------------------
# 4. STYLING
# --------------------------------------------------------------------------
CSS = """
<style>
:root{
  --bg:#F7F6F2; --card:#FFFFFF; --navy:#14213D; --muted:#5B6578;
  --teal:#2F7F7A; --teal-soft:#E3F1EF; --coral:#D9694F; --coral-soft:#FBE9E4;
  --border:#E4E1D8;
}
.stApp{background:var(--bg); color:var(--navy);}
.block-container{padding-top:2rem; padding-bottom:3rem; max-width:1150px;}
h1,h2,h3,h4{color:var(--navy)!important; letter-spacing:-0.01em;}
[data-testid="stSidebar"]{background:#FFFFFF; border-right:1px solid var(--border);}
[data-testid="stSidebar"] .block-container{padding-top:1.2rem;}
div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--card); border-radius:14px; border-color:var(--border)!important;}
.stButton>button{border-radius:10px; border:1px solid var(--border); font-weight:600;}
.stButton>button[kind="primary"]{background:var(--teal); border-color:var(--teal); color:#fff;}
.stButton>button:focus-visible, input:focus-visible{outline:3px solid #8CC5BF!important; outline-offset:1px;}
.brand{display:flex; align-items:center; gap:.7rem; margin-bottom:1rem;}
.brand-logo{width:40px; height:40px; border-radius:12px; background:var(--teal); color:#fff; display:flex; align-items:center; justify-content:center; font-size:1.3rem;}
.brand-name{font-weight:700; font-size:1.15rem; color:var(--navy); line-height:1.1;}
.brand-sub{font-size:.75rem; color:var(--muted);}
.side-disclaimer{font-size:.78rem; color:var(--muted); border-top:1px solid var(--border); padding-top:.8rem; margin-top:1.5rem;}
.hero{background:linear-gradient(135deg,#FFFFFF 0%,#EAF4F2 100%); border:1px solid var(--border); border-radius:18px; padding:1.6rem 1.8rem; margin-bottom:1.2rem;}
.hero h1{margin:0 0 .4rem 0; font-size:2rem;}
.hero p{color:var(--muted); margin:0; font-size:1.02rem; line-height:1.55;}
.card{background:var(--card); border:1px solid var(--border); border-radius:14px; padding:1.1rem 1.25rem; margin-bottom:1rem;}
.card h4{margin:0 0 .5rem 0; font-size:1.02rem;}
.card ul{margin:.3rem 0 0 1.1rem; padding:0;} .card li{margin:.25rem 0; line-height:1.45;}
.card p{margin:.2rem 0; line-height:1.5;}
.muted{color:var(--muted); font-size:.88rem;}
.empty{border:1.5px dashed #C9C6BB; border-radius:14px; padding:1.3rem; text-align:center; color:var(--muted); background:#FCFBF8; margin-bottom:1rem;}
.pill{display:inline-block; padding:.18rem .65rem; border-radius:999px; font-size:.78rem; font-weight:600; background:var(--teal-soft); color:var(--teal);}
.pill.warn{background:var(--coral-soft); color:var(--coral);}
.note{background:var(--coral-soft); border-left:4px solid var(--coral); border-radius:8px; padding:.8rem 1rem; margin:.8rem 0; font-size:.92rem; line-height:1.5;}
.note.info{background:var(--teal-soft); border-left-color:var(--teal);}
.nut{margin:.7rem 0 1rem 0;}
.nut-head{display:flex; justify-content:space-between; font-size:.95rem; margin-bottom:.3rem;}
.bar{height:10px; border-radius:999px; background:#ECEAE3; overflow:hidden;}
.bar-fill{height:100%; border-radius:999px;}
.bar-fill.ok{background:var(--teal);} .bar-fill.warn{background:var(--coral);}
.nut-sub{font-size:.8rem; color:var(--muted); margin-top:.25rem;}
.gauge{position:relative; height:16px; border-radius:999px; background:linear-gradient(90deg,#DCEBE9,#9CC9C4,#2F7F7A); margin:1.1rem 0 .3rem 0;}
.gauge-marker{position:absolute; top:-6px; width:6px; height:28px; border-radius:3px; background:var(--navy); transform:translateX(-50%);}
.gauge-scale{display:flex; justify-content:space-between; font-size:.78rem; color:var(--muted);}
.score-big{font-size:3rem; font-weight:700; color:var(--navy); line-height:1;}
.score-sub{font-size:.85rem; color:var(--muted);}
.footer{font-size:.8rem; color:var(--muted); text-align:center; margin-top:2rem; border-top:1px solid var(--border); padding-top:1rem;}
@media (max-width: 768px){
  .hero{padding:1.1rem;} .hero h1{font-size:1.5rem;} .score-big{font-size:2.3rem;}
  .block-container{padding-left:1rem; padding-right:1rem;}
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 5. MODEL ARTIFACTS (loaded and validated)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    model = tf.keras.models.load_model(BASE_DIR / "anemia_mlp.keras", compile=False)
    scaler = joblib.load(BASE_DIR / "scaler.joblib")
    with open(BASE_DIR / "feature_columns.json", "r", encoding="utf-8") as fh:
        feature_columns = json.load(fh)
    return model, scaler, feature_columns


def validate_artifacts(model, scaler, feature_columns) -> list[str]:
    """Return a list of problems (empty list means everything is consistent)."""
    problems: list[str] = []
    if len(feature_columns) != 17:
        problems.append(f"Expected 17 features, found {len(feature_columns)}.")
    if list(feature_columns) != EXPECTED_FEATURES:
        problems.append(
            "feature_columns.json does not match the expected feature names/order."
        )
    try:
        n_in = model.input_shape[-1]
        if n_in != 17:
            problems.append(f"Model expects {n_in} inputs, not 17.")
    except Exception:  # noqa: BLE001
        problems.append("Could not read the model input shape.")
    n_scaler = getattr(scaler, "n_features_in_", None)
    if n_scaler is not None and n_scaler != 17:
        problems.append(f"Scaler was fitted on {n_scaler} features, not 17.")
    names = getattr(scaler, "feature_names_in_", None)
    if names is not None and list(names) != list(feature_columns):
        problems.append("Scaler feature names/order differ from feature_columns.json.")
    return problems


# --------------------------------------------------------------------------
# 6. SESSION STATE
# --------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "nav": P_DASH,
        "p_age": 21,
        "p_gender": "Female",
        "p_bmi": 22.0,
        "p_diet": "Omnivore",
        "h_cm": 165,
        "w_kg": 60.0,
        "use_manual": False,
        "man_iron": 100.0,
        "man_b12": 100.0,
        "man_folate": 100.0,
        "done_profile": False,
        "done_food": False,
        "done_symptoms": False,
        "result": None,
        "auto_run": False,
        "food_error": False,
    }
    for f in FOODS:
        defaults[f"srv_{f.id}"] = 0
        defaults[f"gm_{f.id}"] = int(f.grams)
    for i in range(FORT_SLOTS):
        defaults[f"fort_name_{i}"] = ""
        defaults[f"fort_srv_{i}"] = 0
        defaults[f"fort_iron_{i}"] = 0.0
        defaults[f"fort_b12_{i}"] = 0.0
        defaults[f"fort_fol_{i}"] = 0.0
    for s in SYMPTOMS:
        defaults[s.key] = False
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    # Keep widget values alive when their page is not currently displayed.
    for key in defaults:
        if key not in ("nav", "result"):
            st.session_state[key] = st.session_state[key]


def goto(page: str) -> None:
    st.session_state["nav"] = page


def mark_done(flag: str) -> None:
    st.session_state[flag] = True


def save_profile_and_next() -> None:
    st.session_state["done_profile"] = True
    st.session_state["nav"] = P_FOOD


def save_food_and_next() -> None:
    if nutrient_state()["has_data"]:
        st.session_state["done_food"] = True
        st.session_state["food_error"] = False
        st.session_state["nav"] = P_SYMP
    else:
        st.session_state["food_error"] = True


def save_symptoms_and_run() -> None:
    st.session_state["done_symptoms"] = True
    st.session_state["auto_run"] = True
    st.session_state["nav"] = P_RESULTS


def first_incomplete_page() -> str:
    ss = st.session_state
    if not ss["done_profile"]:
        return P_PROFILE
    if not (ss["done_food"] or ss["use_manual"]):
        return P_FOOD
    if not ss["done_symptoms"]:
        return P_SYMP
    return P_RESULTS


def apply_bmi() -> None:
    height_m = st.session_state["h_cm"] / 100.0
    bmi = st.session_state["w_kg"] / (height_m ** 2)
    st.session_state["p_bmi"] = float(round(min(80.0, max(5.0, bmi)), 1))


# --------------------------------------------------------------------------
# 7. NUTRIENT CALCULATIONS
# --------------------------------------------------------------------------
def compute_intake() -> tuple[pd.DataFrame, dict[str, float]]:
    """Average daily intake from weekly servings x portion grams x composition."""
    ss = st.session_state
    rows = []
    for f in FOODS:
        servings = ss[f"srv_{f.id}"]
        grams = ss[f"gm_{f.id}"]
        if servings > 0 and grams > 0:
            per_day = servings * grams / 100.0 / 7.0
            rows.append({
                "Category": f.category,
                "Food": f.name,
                "Servings/week": servings,
                "Portion (g)": grams,
                "Iron (mg/day)": f.iron * per_day,
                "Vitamin B12 (µg/day)": f.b12 * per_day,
                "Folate (µg DFE/day)": f.folate * per_day,
            })
    for i in range(FORT_SLOTS):
        servings = ss[f"fort_srv_{i}"]
        iron, b12, fol = ss[f"fort_iron_{i}"], ss[f"fort_b12_{i}"], ss[f"fort_fol_{i}"]
        if servings > 0 and (iron > 0 or b12 > 0 or fol > 0):
            name = ss[f"fort_name_{i}"].strip() or f"Fortified food {i + 1}"
            rows.append({
                "Category": CAT_FORT,
                "Food": name,
                "Servings/week": servings,
                "Portion (g)": np.nan,
                "Iron (mg/day)": iron * servings / 7.0,
                "Vitamin B12 (µg/day)": b12 * servings / 7.0,
                "Folate (µg DFE/day)": fol * servings / 7.0,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        totals = {"iron": 0.0, "b12": 0.0, "folate": 0.0}
    else:
        totals = {
            "iron": float(df["Iron (mg/day)"].sum()),
            "b12": float(df["Vitamin B12 (µg/day)"].sum()),
            "folate": float(df["Folate (µg DFE/day)"].sum()),
        }
    return df, totals


def nutrient_state() -> dict:
    """Combine intake, RDA and percentages (from diary or manual entry)."""
    ss = st.session_state
    rda = reference_rda(int(ss["p_age"]), ss["p_gender"])
    if ss["use_manual"]:
        pct_raw = {
            "iron": float(ss["man_iron"]),
            "b12": float(ss["man_b12"]),
            "folate": float(ss["man_folate"]),
        }
        return {"has_data": True, "source": "manual", "daily": None, "rda": rda,
                "pct_raw": pct_raw, "table": pd.DataFrame()}
    table, totals = compute_intake()
    pct_raw = {k: totals[k] / rda[k] * 100.0 for k in rda}
    return {"has_data": not table.empty, "source": "food diary", "daily": totals,
            "rda": rda, "pct_raw": pct_raw, "table": table}


def nutrient_bars_html(nut: dict) -> str:
    parts = []
    for key, label, unit in NUTRIENTS:
        pct = nut["pct_raw"][key]
        cls = "warn" if pct < 70 else "ok"
        if nut["daily"] is not None:
            sub = f"{nut['daily'][key]:.2f} {unit}/day · RDA reference {nut['rda'][key]:g} {unit}"
        else:
            sub = f"Entered manually · RDA reference {nut['rda'][key]:g} {unit}"
        parts.append(
            f'<div class="nut"><div class="nut-head"><span>{label}</span>'
            f'<strong>{pct:.0f}% of RDA</strong></div>'
            f'<div class="bar"><div class="bar-fill {cls}" style="width:{min(pct, 100):.0f}%"></div></div>'
            f'<div class="nut-sub">{sub}</div></div>'
        )
    return "".join(parts)


# --------------------------------------------------------------------------
# 8. MODEL INPUT AND PREDICTION
# --------------------------------------------------------------------------
def current_symptoms() -> dict[str, int]:
    return {s.key: int(bool(st.session_state[s.key])) for s in SYMPTOMS}


def build_model_input(nut: dict) -> pd.DataFrame:
    ss = st.session_state
    symptoms = current_symptoms()
    diet = ss["p_diet"]
    values = {
        "age": float(ss["p_age"]),
        "gender": float(GENDER_ENCODING[ss["p_gender"]]),
        "bmi": float(ss["p_bmi"]),
        "diet_omnivore": float(diet == "Omnivore"),
        "diet_pescatarian": float(diet == "Pescatarian"),
        "diet_vegan": float(diet == "Vegan"),
        "diet_vegetarian": float(diet == "Vegetarian"),
        # Model inputs are % RDA, limited to the 0–300 range used by the form.
        "iron_percent_rda": float(np.clip(nut["pct_raw"]["iron"], 0, 300)),
        "vitamin_b12_percent_rda": float(np.clip(nut["pct_raw"]["b12"], 0, 300)),
        "folate_percent_rda": float(np.clip(nut["pct_raw"]["folate"], 0, 300)),
        "symptoms_count": float(sum(symptoms.values())),
        **{k: float(v) for k, v in symptoms.items()},
    }
    missing = set(EXPECTED_FEATURES) - set(values)
    extra = set(values) - set(EXPECTED_FEATURES)
    if missing or extra:
        raise ValueError(f"Feature mismatch. Missing: {sorted(missing)}; unexpected: {sorted(extra)}")
    return pd.DataFrame([[values[c] for c in EXPECTED_FEATURES]], columns=EXPECTED_FEATURES)


def run_model(model, scaler, input_df: pd.DataFrame) -> float:
    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled, verbose=0)
    return float(np.asarray(prediction).reshape(-1)[0])


# --------------------------------------------------------------------------
# 9. SMALL UI HELPERS
# --------------------------------------------------------------------------
def card(title: str, body_html: str) -> None:
    st.markdown(f'<div class="card"><h4>{title}</h4>{body_html}</div>', unsafe_allow_html=True)


def empty_state(message: str) -> None:
    st.markdown(f'<div class="empty">{message}</div>', unsafe_allow_html=True)


def disclaimer_note() -> None:
    st.markdown(
        '<div class="note"><strong>Research prototype — not a medical diagnosis.</strong> '
        "This tool cannot diagnose or rule out anemia. Only a blood test and a "
        "professional medical assessment can do that.</div>",
        unsafe_allow_html=True,
    )


def completion() -> tuple[int, int]:
    ss = st.session_state
    nut_ready = ss["use_manual"] or ss["done_food"]
    done = int(ss["done_profile"]) + int(nut_ready) + int(ss["done_symptoms"])
    return done, 3


# --------------------------------------------------------------------------
# 10. SIDEBAR
# --------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-logo">🩸</div><div>'
            '<div class="brand-name">AnemiaSense</div>'
            '<div class="brand-sub">Nutrition &amp; Symptom Assessment</div></div></div>',
            unsafe_allow_html=True,
        )
        st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
        done, total = completion()
        st.caption(f"Assessment progress: {done}/{total} sections")
        st.progress(done / total)
        st.markdown(
            '<div class="side-disclaimer">Research screening tool — not a medical diagnosis.</div>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# 11. PAGES
# --------------------------------------------------------------------------
def page_dashboard() -> None:
    st.markdown(
        '<div class="hero"><h1>Understand your nutrition. Take charge of your health.</h1>'
        "<p>AnemiaSense combines what you eat, how you feel and a machine-learning model "
        "to produce an <strong>experimental screening estimate</strong>. Your food diary is "
        "converted into approximate iron, vitamin B12 and folate intake; together with your "
        "profile and symptoms these become the inputs to a trained neural network. "
        "It is a research prototype, not a diagnosis.</p></div>",
        unsafe_allow_html=True,
    )

    ss = st.session_state
    st.markdown("#### How to use AnemiaSense")
    steps = [
        ("1", "Personal profile", "Enter your age, gender, BMI and dietary pattern.", P_PROFILE, ss["done_profile"]),
        ("2", "Food & nutrition", "Enter what you eat in a typical week.", P_FOOD, ss["done_food"] or ss["use_manual"]),
        ("3", "Symptoms", "Tick any symptoms you currently have.", P_SYMP, ss["done_symptoms"]),
        ("4", "Screening result", "See your risk level and why it was given.", P_RESULTS, ss["result"] is not None),
    ]
    scols = st.columns(4)
    for col, (num, title, desc, _page, is_done) in zip(scols, steps):
        badge = '<span class="pill">Done ✓</span>' if is_done else '<span class="pill warn">To do</span>'
        col.markdown(
            f'<div class="card"><h4>Step {num}: {title}</h4><p class="muted">{desc}</p>{badge}</div>',
            unsafe_allow_html=True,
        )
    nxt = first_incomplete_page()
    label = "Start with Step 1" if nxt == P_PROFILE else "Continue where you left off"
    st.button(label, type="primary", key="start_btn", on_click=goto, args=(nxt,))
    st.caption("After you save each step, the app takes you to the next one automatically.")

    done, total = completion()
    c1, c2 = st.columns([1, 2])
    with c1:
        card("Profile completion", f'<p class="score-big" style="font-size:2.2rem">{done}/{total}</p>'
             '<p class="muted">Profile · Food diary · Symptoms</p>')
    with c2:
        st.markdown("#### Quick access")
        q1, q2, q3 = st.columns(3)
        with q1:
            with st.container(border=True):
                st.markdown("**🥗 Food intake**")
                st.caption("Log weekly foods and portions.")
                st.button("Open food diary", key="q_food", on_click=goto, args=(P_FOOD,))
        with q2:
            with st.container(border=True):
                st.markdown("**🩺 Symptoms**")
                st.caption("Optional symptom checklist.")
                st.button("Open symptoms", key="q_symp", on_click=goto, args=(P_SYMP,))
        with q3:
            with st.container(border=True):
                st.markdown("**📊 Screening**")
                st.caption("Run the experimental model.")
                st.button("Open results", key="q_res", on_click=goto, args=(P_RESULTS,))

    st.markdown("#### Nutrition overview")
    nut = nutrient_state()
    if nut["has_data"]:
        st.markdown(f'<div class="card">{nutrient_bars_html(nut)}'
                    f'<p class="muted">Estimated from your {nut["source"]}. '
                    "RDA is a population reference target; intake below it does not "
                    "by itself mean you are deficient.</p></div>", unsafe_allow_html=True)
    else:
        empty_state("No food data yet. Add foods in <strong>Food &amp; Nutrition</strong> to see "
                    "your estimated iron, vitamin B12 and folate intake.")
    disclaimer_note()


def page_profile() -> None:
    st.header("Personal Profile")
    st.caption("These details are model inputs and are used to choose your reference intake values.")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("Age (years)", min_value=1, max_value=120, step=1, key="p_age",
                            help="Whole years, 1–120.")
        with c2:
            st.radio("Gender", list(GENDER_ENCODING), horizontal=True, key="p_gender",
                     help="Encoded exactly as in the trained model: Female = 1, Male = 0.")
        with c3:
            st.number_input("BMI (kg/m²)", min_value=5.0, max_value=80.0, step=0.1, key="p_bmi",
                            help="Body mass index, 5–80.")
        st.selectbox("Dietary pattern", DIET_OPTIONS, key="p_diet",
                     help="Choose the pattern that best describes your usual diet.")

        with st.expander("Don't know your BMI? Calculate it"):
            b1, b2, b3 = st.columns([1, 1, 1])
            with b1:
                st.number_input("Height (cm)", min_value=50, max_value=250, step=1, key="h_cm")
            with b2:
                st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, step=0.5, key="w_kg")
            with b3:
                st.write("")
                st.button("Use calculated BMI", on_click=apply_bmi)

    age, bmi = st.session_state["p_age"], st.session_state["p_bmi"]
    if age < 18:
        st.warning("The model's training population is not described in the project files. "
                   "Results for people under 18 should be interpreted with particular caution.")
    if bmi < 12 or bmi > 60:
        st.warning("This BMI is unusual — please double-check the value.")
    st.caption("Reference intakes do not account for pregnancy or breastfeeding.")
    st.button("Save and continue to Food & Nutrition →", type="primary", on_click=save_profile_and_next)


def render_food_rows(category: str) -> None:
    h1, h2, h3 = st.columns([3, 1.3, 1.3])
    h1.caption("Food and household measure")
    h2.caption("Servings per week")
    h3.caption("Portion size (g)")
    for f in FOODS_BY_CAT[category]:
        c1, c2, c3 = st.columns([3, 1.3, 1.3])
        c1.markdown(f"**{f.name}**  \n<span class='muted'>{f.measure}</span>", unsafe_allow_html=True)
        c2.number_input(f"{f.name}: servings per week", min_value=0, max_value=70, step=1,
                        key=f"srv_{f.id}", label_visibility="collapsed")
        c3.number_input(f"{f.name}: portion size in grams", min_value=0, max_value=2000, step=5,
                        key=f"gm_{f.id}", label_visibility="collapsed")


def render_fortified() -> None:
    st.caption("Enter values from the product's nutrition label, per serving. "
               "Use µg DFE for folate if the label gives it. Leave at 0 if you don't use fortified foods.")
    for i in range(FORT_SLOTS):
        with st.expander(f"Fortified food {i + 1}", expanded=(i == 0)):
            st.text_input("Food name", key=f"fort_name_{i}", placeholder="e.g. fortified cereal")
            a, b, c, d = st.columns(4)
            a.number_input("Servings per week", min_value=0, max_value=70, step=1, key=f"fort_srv_{i}")
            b.number_input("Iron (mg/serving)", min_value=0.0, max_value=100.0, step=0.1, key=f"fort_iron_{i}")
            c.number_input("B12 (µg/serving)", min_value=0.0, max_value=500.0, step=0.1, key=f"fort_b12_{i}")
            d.number_input("Folate (µg/serving)", min_value=0.0, max_value=2000.0, step=5.0, key=f"fort_fol_{i}")


def page_food() -> None:
    st.header("Food & Nutrition")
    st.caption("Enter your usual weekly intake. Servings and portion sizes are editable; "
               "average daily nutrient intake is calculated from them.")

    tabs = st.tabs(CATEGORIES)
    for tab, category in zip(tabs, CATEGORIES):
        with tab:
            if category == CAT_FORT:
                render_fortified()
            else:
                render_food_rows(category)

    st.divider()
    st.subheader("Estimated average daily intake")
    st.toggle("I'll enter my own % RDA values instead (e.g. from a nutrient analysis)", key="use_manual")
    if st.session_state["use_manual"]:
        m1, m2, m3 = st.columns(3)
        m1.number_input("Iron (% RDA)", min_value=0.0, max_value=300.0, step=5.0, key="man_iron")
        m2.number_input("Vitamin B12 (% RDA)", min_value=0.0, max_value=300.0, step=5.0, key="man_b12")
        m3.number_input("Folate (% RDA)", min_value=0.0, max_value=300.0, step=5.0, key="man_folate")

    nut = nutrient_state()
    if not nut["has_data"]:
        empty_state("No foods entered yet. Set servings per week above (at least one food) to see estimates.")
    else:
        st.markdown(f'<div class="card">{nutrient_bars_html(nut)}</div>', unsafe_allow_html=True)
        if nut["source"] == "food diary":
            table = nut["table"].copy()
            with st.expander("Contribution of each food"):
                st.dataframe(table.round(2), hide_index=True)
            over = [lbl for k, lbl, _ in NUTRIENTS if nut["pct_raw"][k] > 300]
            if over:
                st.info(f"{', '.join(over)} exceeds 300% of RDA; the model input is capped at 300%.")

    with st.expander("Sources and assumptions"):
        st.markdown(
            f"""
- **Food composition:** {FOOD_SOURCE}
- **Reference intakes (RDA):** {RDA_SOURCE} Selected using your age and gender
  (iron {nut['rda']['iron']:g} mg, vitamin B12 {nut['rda']['b12']:g} µg, folate {nut['rda']['folate']:g} µg DFE).
  The RDA basis used when the model was trained is not documented in the project files.
- **Calculation:** daily intake = servings per week × portion (g) ÷ 100 × nutrient per 100 g ÷ 7.
- **Not accounted for:** cooking losses, recipe differences, added water in dal/curries, iron bioavailability
  (plant iron is absorbed less well than animal iron), absorption enhancers or inhibitors,
  supplements, and pregnancy or breastfeeding.
- Food-frequency estimates are approximate. They do **not** measure iron, B12 or folate status.
            """
        )
    if st.session_state["food_error"]:
        st.warning("Please enter at least one food (servings per week above 0) before continuing.")
    st.button("Save and continue to Symptoms →", type="primary", on_click=save_food_and_next)


def page_symptoms() -> None:
    st.header("Symptom Assessment")
    st.markdown(
        '<div class="note info">These symptoms can have many causes and are <strong>not specific to '
        "anemia</strong>. Select only those you currently experience; all are optional.</div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        c1, c2 = st.columns(2)
        for i, s in enumerate(SYMPTOMS):
            with (c1 if i % 2 == 0 else c2):
                st.checkbox(s.label, key=s.key, help=s.hint)
    count = sum(current_symptoms().values())
    st.caption(f"Symptoms selected: {count}")
    st.caption("You can continue without selecting any symptoms.")
    st.button("Save and see my screening result →", type="primary", on_click=save_symptoms_and_run)


def risk_level(score: float) -> tuple[str, str, str]:
    """Return (headline, css class, short meaning) for a model score."""
    if abs(score - DECISION_THRESHOLD) <= BORDERLINE_BAND:
        return ("Borderline screening result", "warn",
                "Your inputs sit close to the model's cut-off, so the model is uncertain.")
    if score >= DECISION_THRESHOLD:
        return ("Higher anemia-risk pattern", "warn",
                "Your inputs resemble the anemia cases the model was trained on.")
    return ("Lower anemia-risk pattern", "",
            "Your inputs resemble the non-anemia cases the model was trained on.")


FEATURE_GROUPS = [
    ("Iron intake", ["iron_percent_rda"]),
    ("Vitamin B12 intake", ["vitamin_b12_percent_rda"]),
    ("Folate intake", ["folate_percent_rda"]),
    ("Symptoms", ["symptoms_count"] + [s.key for s in SYMPTOMS]),
    ("Age", ["age"]),
    ("BMI", ["bmi"]),
    ("Gender", ["gender"]),
    ("Dietary pattern", ["diet_omnivore", "diet_pescatarian", "diet_vegan", "diet_vegetarian"]),
]


def explain_score(model, scaler, input_df: pd.DataFrame, score: float) -> list[dict]:
    """
    Local explanation by ablation: for each group of inputs, replace it with the
    training-data average (scaler.mean_) and see how the model score changes.
    Positive effect = this input pushed the score up for this person.
    """
    means = getattr(scaler, "mean_", None)
    if means is None:
        return []
    mean_by_col = dict(zip(EXPECTED_FEATURES, [float(m) for m in means]))
    rows = []
    for _, cols in FEATURE_GROUPS:
        altered = input_df.iloc[0].copy()
        for c in cols:
            altered[c] = mean_by_col[c]
        rows.append(altered)
    batch = pd.DataFrame(rows, columns=EXPECTED_FEATURES)
    preds = np.asarray(model.predict(scaler.transform(batch), verbose=0)).reshape(-1)
    eps = 1e-6

    def logit(v: float) -> float:
        v = float(np.clip(v, eps, 1 - eps))
        return float(np.log(v / (1 - v)))

    # Log-odds are used so effects stay visible even when the score is close to 0 or 1.
    base = logit(score)
    out = []
    for (label, _cols), p in zip(FEATURE_GROUPS, preds):
        out.append({"label": label, "effect": base - logit(p)})
    return sorted(out, key=lambda r: abs(r["effect"]), reverse=True)


def describe_group(label: str, row: pd.Series, symptoms: list[str]) -> str:
    if label == "Iron intake":
        return f"{row['iron_percent_rda']:.0f}% of RDA"
    if label == "Vitamin B12 intake":
        return f"{row['vitamin_b12_percent_rda']:.0f}% of RDA"
    if label == "Folate intake":
        return f"{row['folate_percent_rda']:.0f}% of RDA"
    if label == "Symptoms":
        return f"{len(symptoms)} selected" + (f" ({', '.join(symptoms)})" if symptoms else "")
    if label == "Age":
        return f"{row['age']:.0f} years"
    if label == "BMI":
        return f"{row['bmi']:.1f}"
    if label == "Gender":
        return "Female" if row["gender"] >= 0.5 else "Male"
    if label == "Dietary pattern":
        for name, col in [("Omnivore", "diet_omnivore"), ("Pescatarian", "diet_pescatarian"),
                          ("Vegan", "diet_vegan"), ("Vegetarian", "diet_vegetarian")]:
            if row[col] >= 0.5:
                return name
    return ""


def compute_and_store_result(model, scaler, nut: dict, input_df: pd.DataFrame) -> None:
    score = run_model(model, scaler, input_df)
    try:
        drivers = explain_score(model, scaler, input_df, score)
    except Exception:  # noqa: BLE001
        drivers = []
    st.session_state["result"] = {
        "score": score,
        "sig": tuple(input_df.iloc[0].tolist()),
        "input": input_df.copy(),
        "drivers": drivers,
        "time": datetime.now().strftime("%d %b %Y, %H:%M"),
        "nut": {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in nut.items()},
        "profile": {
            "Age": st.session_state["p_age"], "Gender": st.session_state["p_gender"],
            "BMI": st.session_state["p_bmi"], "Diet": st.session_state["p_diet"],
        },
        "symptoms": [s.label for s in SYMPTOMS if st.session_state[s.key]],
    }


def page_results(model, scaler) -> None:
    st.header("Screening Results")
    disclaimer_note()

    nut = nutrient_state()
    ready = nut["has_data"]
    if not ready:
        empty_state("Complete <strong>Food &amp; Nutrition</strong> first (or enter % RDA values manually) "
                    "to get your screening result.")
        st.button("Go to Food & Nutrition", on_click=goto, args=(P_FOOD,))
        return

    try:
        input_df = build_model_input(nut)
    except Exception as err:  # noqa: BLE001
        st.error(f"Could not prepare the model input: {err}")
        return
    signature = tuple(input_df.iloc[0].tolist())

    # Run automatically after the symptom step, or when the user presses the button.
    if st.session_state["auto_run"]:
        st.session_state["auto_run"] = False
        try:
            compute_and_store_result(model, scaler, nut, input_df)
        except Exception as err:  # noqa: BLE001
            st.error(f"Could not calculate the estimate: {err}")
    elif st.button("Run / refresh screening", type="primary"):
        try:
            compute_and_store_result(model, scaler, nut, input_df)
        except Exception as err:  # noqa: BLE001
            st.error(f"Could not calculate the estimate: {err}")

    result = st.session_state.get("result")
    if not result:
        empty_state("Press <strong>Run / refresh screening</strong> to see your result.")
        return
    if result["sig"] != signature:
        st.warning("Your inputs have changed since this result was calculated. "
                   "Press 'Run / refresh screening' to update it.")

    score = result["score"]
    headline, cls, meaning = risk_level(score)
    marker = float(np.clip(score, 0.0, 1.0)) * 100
    threshold_pos = DECISION_THRESHOLD * 100

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(
            f'<div class="card"><h4>Your screening risk level '
            f'<span class="pill warn">Experimental</span></h4>'
            f'<div class="score-big" style="font-size:1.9rem">{headline}</div>'
            f'<p class="muted" style="margin-top:.4rem">{meaning}</p>'
            f'<div class="gauge"><div class="gauge-marker" style="left:{marker:.1f}%"></div></div>'
            f'<div class="gauge-scale"><span>0 lower</span><span>model cut-off {threshold_pos:.0f}</span>'
            f'<span>100 higher</span></div>'
            f'<p class="score-sub" style="margin-top:.6rem">Model score <strong>{score * 100:.1f} / 100</strong> '
            f'· calculated {result["time"]}</p></div>',
            unsafe_allow_html=True,
        )
    with right:
        profile = result["profile"]
        symptoms = result["symptoms"]
        sym_html = "".join(f"<li>{html.escape(s)}</li>" for s in symptoms) or "<li>None selected</li>"
        card(
            "Summary of your answers",
            f"<p>Age <strong>{profile['Age']}</strong> · {profile['Gender']} · "
            f"BMI <strong>{profile['BMI']}</strong> · {profile['Diet']}</p>"
            f"<p><strong>Symptoms ({len(symptoms)}):</strong></p><ul>{sym_html}</ul>",
        )

    st.markdown("### Why did you get this result?")
    row = result["input"].iloc[0]
    drivers = result.get("drivers", [])
    meaningful = [d for d in drivers if abs(d["effect"]) >= 0.3]
    if not meaningful:
        st.info("No single answer changed the model score noticeably; the result reflects your answers taken together.")
    else:
        items = []
        for d in meaningful[:5]:
            value = describe_group(d["label"], row, result["symptoms"])
            direction = "raised" if d["effect"] > 0 else "lowered"
            size = abs(d["effect"])
            strength = "strongly" if size >= 2 else ("clearly" if size >= 1 else "slightly")
            items.append(
                f"<li><strong>{d['label']}</strong> ({html.escape(value)}) "
                f"<strong>{strength} {direction}</strong> the anemia-risk score.</li>"
            )
        st.markdown(
            '<div class="card"><h4>What influenced the model most</h4><ul>' + "".join(items) + "</ul>"
            '<p class="muted">Each effect is measured by replacing that answer with the average of the '
            "training data and seeing how the model's score changes (on a log-odds scale). "
            "Effects interact, so they do not add up exactly. "
            "This explains how the <em>model</em> behaves; it does not prove a medical cause.</p></div>",
            unsafe_allow_html=True,
        )

    facts = []
    for key, label, unit in NUTRIENTS:
        pct = result["nut"]["pct_raw"][key]
        if pct < 70:
            facts.append(f"<li><strong>{label}</strong> intake is low at {pct:.0f}% of the RDA reference.</li>")
        elif pct < 100:
            facts.append(f"<li><strong>{label}</strong> intake is slightly below the RDA reference ({pct:.0f}%).</li>")
    if result["symptoms"]:
        facts.append(f"<li>You reported {len(result['symptoms'])} symptom(s), which can have many causes "
                     "and are not specific to anemia.</li>")
    if profile["Diet"] in ("Vegan", "Vegetarian"):
        facts.append(f"<li>A {profile['Diet'].lower()} diet can make iron and vitamin B12 harder to get "
                     "in sufficient amounts, so these are worth watching.</li>")
    if not facts:
        facts.append("<li>None of your answers stood out as a clear risk factor "
                     "(intakes at or above reference, no symptoms reported).</li>")
    st.markdown('<div class="card"><h4>What stands out in your answers</h4><ul>' + "".join(facts) + "</ul></div>",
                unsafe_allow_html=True)

    st.markdown(f'<div class="card"><h4>Estimated nutrient intake</h4>{nutrient_bars_html(result["nut"])}'
                f'<p class="muted">Source: {result["nut"]["source"]}. Percentages are relative to the '
                "RDA reference, not a measure of nutrient status.</p></div>", unsafe_allow_html=True)

    if score >= DECISION_THRESHOLD - BORDERLINE_BAND:
        advice = ("Because the model flagged a higher or borderline pattern, please consider seeing a "
                  "healthcare professional and asking about a complete blood count (CBC).")
    else:
        advice = ("A lower score does <strong>not</strong> rule out anemia. If you have symptoms such as "
                  "persistent tiredness, pale skin or tingling, please still see a healthcare professional.")
    st.markdown(
        '<div class="note info"><strong>What to do next.</strong> ' + advice +
        " Only laboratory tests (hemoglobin, and where appropriate ferritin, vitamin B12 or folate) can show "
        "whether anemia is present and why. This tool is a research prototype and not a diagnosis. "
        "Do not use it to delay medical care.</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Exact features sent to the model (17 features, in model order)"):
        st.dataframe(result["input"], hide_index=True)
    st.button("See dietary suggestions →", on_click=goto, args=(P_RECS,))


def build_recommendations(diet: str, nut: dict, symptom_count: int) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    pct = nut["pct_raw"] if nut["has_data"] else None
    low = lambda k: pct is not None and pct[k] < 100  # noqa: E731

    plant_iron = ("Lentils/dal, chickpeas (chana), kidney beans (rajma) and moong, plus leafy greens such as "
                  "palak (spinach) and chaulai (amaranth leaves).")
    vit_c = ("Pair plant-based iron foods with vitamin C-rich foods in the same meal: lemon, amla, guava, "
             "orange, tomato or capsicum.")

    if low("iron") or pct is None:
        items = [plant_iron, vit_c,
                 "Tea and coffee with meals can reduce absorption of plant iron; try having them between meals."]
        if diet == "Omnivore":
            items.insert(1, "Eggs, chicken, mutton and fish provide iron that is absorbed more readily.")
        elif diet == "Pescatarian":
            items.insert(1, "Fish (such as sardines) and eggs, if you eat them, add iron.")
        elif diet == "Vegetarian":
            items.insert(1, "Eggs can add iron if they are part of your vegetarian pattern.")
        sections.append(("Iron", items))

    if low("b12") or pct is None:
        if diet == "Vegan":
            items = ["Plant foods do not reliably contain vitamin B12. Look for B12-fortified foods "
                     "(some plant milks, cereals or nutritional yeast) and check the label.",
                     "Consider discussing B12 testing, and what is appropriate for you, with a doctor or dietitian."]
        elif diet == "Vegetarian":
            items = ["Milk, curd and other dairy foods, and eggs if you eat them, contain B12.",
                     "B12-fortified foods can help; check the label."]
        elif diet == "Pescatarian":
            items = ["Fish (sardines, salmon and others), dairy foods and eggs are B12 sources.",
                     "B12-fortified foods can help; check the label."]
        else:
            items = ["Milk, curd, eggs, fish, chicken and mutton all provide B12.",
                     "B12-fortified foods can help where your intake is low; check the label."]
        sections.append(("Vitamin B12", items))

    if low("folate") or pct is None:
        sections.append(("Folate", [
            "Dals, chickpeas and rajma are excellent folate sources.",
            "Leafy greens (palak, chaulai) and fruits such as guava, papaya and orange contribute folate.",
            "Light cooking or steaming preserves more folate than long boiling.",
        ]))

    if pct is not None and not sections:
        sections.append(("Keep it up", [
            "Your estimated intake of all three nutrients is at or above the reference values. "
            "Keep a varied diet with pulses, greens, fruit and, where suitable, dairy, eggs, fish or meat."
        ]))
    return sections


def page_recs() -> None:
    st.header("Dietary Recommendations")
    st.markdown(
        '<div class="note info"><strong>General food guidance only.</strong> These suggestions are not '
        "medical treatment. Dietary changes alone do not treat anemia, and the cause of anemia needs "
        "professional evaluation. A clinician decides on any treatment, including whether supplements "
        "are appropriate.</div>",
        unsafe_allow_html=True,
    )
    nut = nutrient_state()
    diet = st.session_state["p_diet"]
    symptom_count = sum(current_symptoms().values())

    if not nut["has_data"]:
        empty_state("Enter your food diary to personalise these suggestions. "
                    "Showing general guidance for your dietary pattern.")
        st.button("Go to food diary", on_click=goto, args=(P_FOOD,))
    else:
        parts = []
        for key, label, _ in NUTRIENTS:
            pct = nut["pct_raw"][key]
            if pct < 70:
                parts.append(f"{label}: <strong>notable gap</strong> ({pct:.0f}% of RDA)")
            elif pct < 100:
                parts.append(f"{label}: slightly below reference ({pct:.0f}%)")
            else:
                parts.append(f"{label}: meets reference ({pct:.0f}%)")
        card("Your estimated dietary gaps", "<ul>" + "".join(f"<li>{p}</li>" for p in parts) + "</ul>")

    st.markdown(f"**Dietary pattern:** {diet}")
    for title, items in build_recommendations(diet, nut, symptom_count):
        card(title, "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")

    if symptom_count:
        st.markdown(
            '<div class="note">You reported symptoms. Please consider seeing a healthcare professional '
            "for evaluation rather than relying on diet changes.</div>",
            unsafe_allow_html=True,
        )


def page_about(model, scaler, feature_columns) -> None:
    st.header("About the Model")
    st.markdown(
        '<div class="note"><strong>Experimental research prototype.</strong> No clinical validation or '
        "diagnostic accuracy is claimed.</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        try:
            params = f"{model.count_params():,}"
            layers = ", ".join(layer.__class__.__name__ for layer in model.layers)
        except Exception:  # noqa: BLE001
            params, layers = "n/a", "n/a"
        card("Model", f"<ul><li><strong>Type:</strong> Multilayer Perceptron (MLP), TensorFlow/Keras</li>"
                      f"<li><strong>Inputs:</strong> 17 features from the existing training pipeline</li>"
                      f"<li><strong>Input shape:</strong> {model.input_shape}</li>"
                      f"<li><strong>Parameters:</strong> {params}</li>"
                      f"<li><strong>Layers:</strong> {html.escape(layers)}</li></ul>")
    with c2:
        card("Preprocessing", f"<ul><li>Saved <strong>{scaler.__class__.__name__}</strong> "
                              "(scaler.joblib) applied to all 17 features in the saved order.</li>"
                              "<li>The model and scaler are not retrained or modified by this app.</li></ul>")

    st.subheader("Feature schema (in model order)")
    schema = pd.DataFrame({"#": range(1, len(feature_columns) + 1), "Feature": feature_columns})
    st.dataframe(schema, hide_index=True)

    card("Interpreting the score",
         "<ul><li>The score is the model's raw sigmoid output (0–1), shown on a 0–100 scale.</li>"
         "<li>The training notebook used <strong>0.5</strong> as the cut-off to report precision and recall. "
         "This app shows scores within 0.10 of that cut-off as <em>borderline</em>; that band is a presentation "
         "choice, not a validated rule.</li>"
         "<li>The score is not a calibrated probability and cannot diagnose or exclude anemia.</li></ul>")
    card("Training summary (from the training notebook)",
         "<ul><li>Dataset: 2,754 rows (1,509 labelled not anemic, 1,245 labelled anemic), 17 non-blood features.</li>"
         "<li>Split: 80% train / 20% test (stratified); scaler fitted on the training portion only.</li>"
         "<li>Network: 64 → 32 → 16 → 1 (ReLU, dropout 0.30 and 0.20, sigmoid output), 3,777 parameters.</li>"
         "<li>Reported test results: accuracy 0.976, ROC-AUC 0.996 (held-out 551 rows from the same dataset).</li></ul>")
    card("Limitations",
         "<ul><li>The notebook does not say where <code>anemia_label</code> came from (for example laboratory "
         "hemoglobin versus a rule or simulation), nor how the dataset and the % RDA values were produced.</li>"
         "<li>Near-perfect scores from non-blood features alone are unusual in real clinical data. They can "
         "happen when labels are derived from the same inputs or the data are synthetic, so the provenance "
         "should be checked before drawing conclusions.</li>"
         "<li>The test set comes from the same dataset, so these metrics do not establish clinical validity, "
         "calibration or generalisation to new people.</li>"
         "<li>Food-based nutrient estimates are approximate and ignore cooking, bioavailability and supplements.</li>"
         "<li>Symptoms are self-reported and non-specific.</li>"
         "<li>Reference intakes assume non-pregnant, non-lactating individuals.</li></ul>")
    card("Data sources",
         f"<ul><li><strong>Food composition:</strong> {html.escape(FOOD_SOURCE)}</li>"
         f"<li><strong>Reference intakes:</strong> {html.escape(RDA_SOURCE)}</li></ul>")


# --------------------------------------------------------------------------
# 12. MAIN
# --------------------------------------------------------------------------
def main() -> None:
    inject_css()
    try:
        model, scaler, feature_columns = load_artifacts()
    except Exception as err:  # noqa: BLE001
        st.error(f"Could not load the model files from {BASE_DIR}: {err}")
        st.stop()

    problems = validate_artifacts(model, scaler, feature_columns)
    if problems:
        st.error("Model artifact validation failed:\n\n- " + "\n- ".join(problems))
        st.stop()

    init_state()
    render_sidebar()

    page = st.session_state["nav"]
    if page == P_DASH:
        page_dashboard()
    elif page == P_PROFILE:
        page_profile()
    elif page == P_FOOD:
        page_food()
    elif page == P_SYMP:
        page_symptoms()
    elif page == P_RESULTS:
        page_results(model, scaler)
    elif page == P_RECS:
        page_recs()
    else:
        page_about(model, scaler, feature_columns)

    st.markdown(
        '<div class="footer">Research prototype • Not a medical diagnosis • '
        "Do not use this result to delay medical care</div>",
        unsafe_allow_html=True,
    )


main()
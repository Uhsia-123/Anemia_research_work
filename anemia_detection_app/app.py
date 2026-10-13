
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Anemia Screening Assistant",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Anemia Screening Assistant")
st.write(
    "Explore your dietary intake, report symptoms, and receive "
    "an experimental machine-learning screening estimate."
)

st.warning(
    "This tool is for educational and research purposes only. "
    "It cannot diagnose or rule out anemia. A blood test and "
    "professional medical assessment are needed for diagnosis."
)


# --------------------------------------------------
# 2. LOAD THE SAVED MODEL AND PREPROCESSOR
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        BASE_DIR / "anemia_mlp.keras",
        compile=False,
    )
    scaler = joblib.load(BASE_DIR / "scaler.joblib")

    with open(BASE_DIR / "feature_columns.json", "r") as file:
        feature_columns = json.load(file)

    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_artifacts()
except Exception as error:
    st.error(f"Could not load the model files: {error}")
    st.stop()


if len(feature_columns) != 17:
    st.error(
        f"Expected 17 model features, but found "
        f"{len(feature_columns)}. Check feature_columns.json."
    )
    st.stop()


# --------------------------------------------------
# 3. USER PROFILE
# --------------------------------------------------

st.header("1. Your Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=21,
    )

with col2:
    gender_choice = st.selectbox(
        "Gender",
        ["Female", "Male"],
    )

with col3:
    bmi = st.number_input(
        "BMI (kg/m²)",
        min_value=5.0,
        max_value=80.0,
        value=22.0,
        step=0.1,
    )

gender = 1 if gender_choice == "Female" else 0


# --------------------------------------------------
# 4. DIETARY PATTERN
# --------------------------------------------------

st.header("2. Dietary Pattern")

diet_choice = st.selectbox(
    "Which dietary pattern best describes you?",
    [
        "Omnivore",
        "Vegetarian",
        "Vegan",
        "Pescatarian",
    ],
)

diet_features = {
    "diet_omnivore": int(diet_choice == "Omnivore"),
    "diet_pescatarian": int(diet_choice == "Pescatarian"),
    "diet_vegan": int(diet_choice == "Vegan"),
    "diet_vegetarian": int(diet_choice == "Vegetarian"),
}


# --------------------------------------------------
# 5. FOOD FREQUENCY QUESTIONNAIRE
# --------------------------------------------------

st.header("3. Weekly Food Questionnaire")

st.caption(
    "Enter your usual intake. These questions help estimate dietary "
    "nutrients; they do not directly establish your iron or vitamin status."
)

st.subheader("Iron-containing foods")

col1, col2 = st.columns(2)

with col1:
    meat_servings = st.number_input(
        "Meat servings per week",
        min_value=0,
        max_value=42,
        value=2,
    )

    egg_servings = st.number_input(
        "Egg servings per week",
        min_value=0,
        max_value=42,
        value=3,
    )

with col2:
    beans_servings = st.number_input(
        "Beans, lentils or chickpeas servings per week",
        min_value=0,
        max_value=42,
        value=4,
    )

    leafy_green_servings = st.number_input(
        "Leafy green vegetable servings per week",
        min_value=0,
        max_value=42,
        value=4,
    )

st.subheader("Vitamin B12 and folate sources")

col1, col2 = st.columns(2)

with col1:
    dairy_servings = st.number_input(
        "Milk or dairy servings per week",
        min_value=0,
        max_value=42,
        value=7,
    )

    fish_servings = st.number_input(
        "Fish servings per week",
        min_value=0,
        max_value=42,
        value=1,
    )

with col2:
    fortified_servings = st.number_input(
        "B12-fortified food servings per week",
        min_value=0,
        max_value=42,
        value=0,
    )

    fruit_servings = st.number_input(
        "Fruit servings per week",
        min_value=0,
        max_value=42,
        value=7,
    )


# --------------------------------------------------
# 6. NUTRIENT ESTIMATES
# --------------------------------------------------

st.subheader("Estimated nutrient intake")

st.info(
    "The current model requires iron, vitamin B12 and folate intake "
    "as percentages of reference dietary allowances. Food frequency "
    "alone is not enough to calculate reliable values without defined "
    "portion sizes and food-composition data. For now, enter your "
    "estimated percentages below if you already have them."
)

col1, col2, col3 = st.columns(3)

with col1:
    iron_percent_rda = st.number_input(
        "Estimated iron (% RDA)",
        min_value=0.0,
        max_value=300.0,
        value=100.0,
        step=5.0,
    )

with col2:
    b12_percent_rda = st.number_input(
        "Estimated vitamin B12 (% RDA)",
        min_value=0.0,
        max_value=300.0,
        value=100.0,
        step=5.0,
    )

with col3:
    folate_percent_rda = st.number_input(
        "Estimated folate (% RDA)",
        min_value=0.0,
        max_value=300.0,
        value=100.0,
        step=5.0,
    )


# --------------------------------------------------
# 7. SYMPTOM QUESTIONNAIRE
# --------------------------------------------------

st.header("4. Symptoms")

st.caption(
    "These symptoms can have many causes and are not specific to anemia."
)

symptoms = {
    "has_fatigue": st.checkbox("Fatigue or unusual tiredness"),
    "has_muscle_weakness": st.checkbox("Muscle weakness"),
    "has_numbness_tingling": st.checkbox("Numbness or tingling"),
    "has_memory_problems": st.checkbox("Memory or concentration problems"),
    "has_pale_skin": st.checkbox("Paler skin than usual"),
    "has_bone_pain": st.checkbox("Bone pain"),
}

symptoms_count = sum(symptoms.values())


# --------------------------------------------------
# 8. PREPARE THE MODEL INPUT
# --------------------------------------------------

st.header("5. Screening Estimate")


def build_model_input():
    input_values = {
        "age": float(age),
        "gender": float(gender),
        "bmi": float(bmi),
        **diet_features,
        "iron_percent_rda": float(iron_percent_rda),
        "vitamin_b12_percent_rda": float(b12_percent_rda),
        "folate_percent_rda": float(folate_percent_rda),
        "symptoms_count": float(symptoms_count),
        **{key: int(value) for key, value in symptoms.items()},
    }

    missing = set(feature_columns) - set(input_values)
    extra = set(input_values) - set(feature_columns)

    if missing or extra:
        raise ValueError(
            f"Feature mismatch. Missing: {sorted(missing)}; "
            f"unexpected: {sorted(extra)}"
        )

    input_df = pd.DataFrame(
        [[input_values[column] for column in feature_columns]],
        columns=feature_columns,
    )

    return input_df


if st.button("Calculate Screening Estimate", type="primary"):
    try:
        input_df = build_model_input()
        scaled_input = scaler.transform(input_df)

        prediction = model.predict(
            scaled_input,
            verbose=0,
        )

        score = float(np.asarray(prediction).reshape(-1)[0])

        st.subheader("Model output")

        st.metric(
            "Model score",
            f"{score * 100:.1f}%",
        )

        st.progress(float(np.clip(score, 0.0, 1.0)))

        st.caption(
            "This is the model's sigmoid output, not a confirmed "
            "probability of having anemia. Its clinical meaning and "
            "decision threshold have not been established here."
        )

        st.write("**Features submitted to the model:**")
        st.dataframe(input_df, use_container_width=True)

        st.info(
            "If you have symptoms or are concerned about anemia, "
            "consult a healthcare professional. A complete blood count "
            "and other tests may be needed to identify the cause."
        )

    except Exception as error:
        st.error(f"Could not calculate the estimate: {error}")


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "Research prototype • Not a medical diagnosis • "
    "Do not use this result to delay medical care"
)
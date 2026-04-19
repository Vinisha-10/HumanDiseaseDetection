# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 02:02:06 2025

@author: Vinisha Singh
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Human Disease Detection",
    page_icon="+",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_models():
    return {
        "Decision Tree": pickle.load(open(os.path.join(BASE_DIR, "artifacts", "DT_model.sav"), "rb")),
        "KNN": pickle.load(open(os.path.join(BASE_DIR, "artifacts", "knn_model.sav"), "rb")),
        "Naive Bayes": pickle.load(open(os.path.join(BASE_DIR, "artifacts", "NBayes.sav"), "rb")),
        "Random Forest": pickle.load(open(os.path.join(BASE_DIR, "artifacts", "Rforest.sav"), "rb")),
        "SVM": pickle.load(open(os.path.join(BASE_DIR, "artifacts", "svm_model.pkl"), "rb")),
    }


SYMPTOMS = [
    "back_pain", "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine",
    "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", "swelling_of_stomach",
    "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision", "phlegm", "throat_irritation",
    "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", "chest_pain", "weakness_in_limbs",
    "fast_heart_rate", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool",
    "irritation_in_anus", "neck_pain", "dizziness", "cramps", "bruising", "obesity", "swollen_legs",
    "swollen_blood_vessels", "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails",
    "swollen_extremeties", "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips",
    "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints",
    "movement_stiffness", "spinning_movements", "loss_of_balance", "unsteadiness",
    "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort", "foul_smell_of urine",
    "continuous_feel_of_urine", "passage_of_gases", "internal_itching", "toxic_look_(typhos)",
    "depression", "irritability", "muscle_pain", "altered_sensorium", "red_spots_over_body", "belly_pain",
    "abnormal_menstruation", "dischromic _patches", "watering_from_eyes", "increased_appetite", "polyuria",
    "family_history", "mucoid_sputum", "rusty_sputum", "lack_of_concentration", "visual_disturbances",
    "receiving_blood_transfusion", "receiving_unsterile_injections", "coma", "stomach_bleeding",
    "distention_of_abdomen", "history_of_alcohol_consumption", "fluid_overload", "blood_in_sputum",
    "prominent_veins_on_calf", "palpitations", "painful_walking", "pus_filled_pimples", "blackheads",
    "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails",
    "blister", "red_sore_around_nose", "yellow_crust_ooze",
]

DISEASES = [
    "Fungal infection", "Allergy", "GERD", "Chronic cholestasis", "Drug Reaction", "Peptic ulcer disease",
    "AIDS", "Diabetes", "Gastroenteritis", "Bronchial Asthma", "Hypertension", "Migraine",
    "Cervical spondylosis", "Paralysis (brain hemorrhage)", "Jaundice", "Malaria", "Chicken pox",
    "Dengue", "Typhoid", "Hepatitis A", "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E",
    "Alcoholic hepatitis", "Tuberculosis", "Common Cold", "Pneumonia", "Dimorphic hemorrhoids (piles)",
    "Heart attack", "Varicose veins", "Hypothyroidism", "Hyperthyroidism", "Hypoglycemia",
    "Osteoarthritis", "Arthritis", "(Vertigo) Paroxysmal Positional Vertigo", "Acne",
    "Urinary tract infection", "Psoriasis", "Impetigo",
]


def pretty_label(symptom):
    return symptom.replace("_", " ").strip().title()


def build_feature_vector(symptoms):
    return np.array([[1 if symptom in symptoms else 0 for symptom in SYMPTOMS]])


def normalize_prediction(raw_prediction):
    if isinstance(raw_prediction, np.ndarray):
        raw_prediction = raw_prediction[0]

    if isinstance(raw_prediction, (np.integer, int)):
        index = int(raw_prediction)
        return DISEASES[index] if 0 <= index < len(DISEASES) else "Not Found"

    return str(raw_prediction).strip()


def predict_disease(model, symptoms):
    input_vector = build_feature_vector(symptoms)
    prediction = model.predict(input_vector)[0]
    return normalize_prediction(prediction)


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(123, 211, 234, 0.18), transparent 32%),
                radial-gradient(circle at top right, rgba(167, 139, 250, 0.14), transparent 26%),
                linear-gradient(180deg, #f4f8fb 0%, #eaf1f7 100%);
            color: #17324d;
        }
        .hero-card, .panel-card {
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(34, 77, 122, 0.12);
            border-radius: 22px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 18px 45px rgba(31, 65, 104, 0.10);
            backdrop-filter: blur(8px);
        }
        .hero-kicker {
            display: inline-block;
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
            background: #e1f0fb;
            color: #155e8a;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .hero-title {
            margin: 0.75rem 0 0.35rem 0;
            color: #102a43;
            font-size: 2.4rem;
            font-weight: 800;
            line-height: 1.1;
        }
        .hero-copy {
            color: #486581;
            font-size: 1rem;
            margin-bottom: 0.35rem;
        }
        .author-line {
            color: #0f766e;
            font-weight: 700;
            margin-top: 0.6rem;
        }
        .pill {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            margin: 0.2rem 0.25rem 0.1rem 0;
            border-radius: 999px;
            background: #edf6ff;
            color: #1d4f78;
            border: 1px solid #c9e3f6;
            font-size: 0.88rem;
        }
        h1, h2, h3, .stSubheader, .stHeader {
            color: #102a43 !important;
        }
        p, label, .stMarkdown, .stCaption, .stText, .stInfo {
            color: #334e68;
        }
        [data-testid="stMetricValue"] {
            color: #0b6e99;
        }
        [data-testid="stMetricLabel"] {
            color: #486581;
        }
        .stButton > button {
            background: linear-gradient(135deg, #0b6e99 0%, #0f766e 100%);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            font-weight: 700;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #09577a 0%, #0b5f59 100%);
            color: #ffffff;
        }
        [data-baseweb="select"] > div {
            background-color: #f8fbfe;
            border-color: #c7d9e8;
            color: #17324d;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #153a5b 0%, #1f4f78 100%);
        }
        [data-testid="stSidebar"] * {
            color: #f4f8fb !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    inject_styles()
    models = load_models()

    with st.sidebar:
        st.header("How It Works")
        st.write("Select between 2 and 5 symptoms, then compare predictions from all five trained models.")
        st.info("This tool supports screening exploration only and should not replace medical advice.")
        st.caption("Author: Vinisha Singh")

    st.markdown(
        """
        <div class="hero-card">
            <span class="hero-kicker">ML-Powered Symptom Screening</span>
            <div class="hero-title">Human Disease Detection Dashboard</div>
            <div class="hero-copy">
                Explore likely disease predictions from Decision Tree, KNN, Naive Bayes, Random Forest,
                and SVM models in a cleaner, more visual interface.
            </div>
            <div class="author-line">Designed by Vinisha Singh</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Models Available", len(models))
    metric_col2.metric("Symptom Library", len(SYMPTOMS))
    metric_col3.metric("Symptoms Required", "2 minimum")

    left_col, right_col = st.columns([1.2, 0.8], gap="large")

    with left_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Select Symptoms")
        selected_symptoms = st.multiselect(
            "Choose up to 5 symptoms",
            options=SYMPTOMS,
            format_func=pretty_label,
            max_selections=5,
            placeholder="Start typing symptoms such as chest_pain or mild_fever",
        )
        st.caption("Pick at least two symptoms to generate predictions.")
        predict_clicked = st.button("Analyze Symptoms", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Selected Snapshot")
        if selected_symptoms:
            st.markdown(
                "".join(f'<span class="pill">{pretty_label(symptom)}</span>' for symptom in selected_symptoms),
                unsafe_allow_html=True,
            )
        else:
            st.write("No symptoms selected yet.")
        st.write("Use the selector on the left to build a small symptom profile and compare model agreement.")
        st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked:
        if len(selected_symptoms) < 2:
            st.error("Please select at least two symptoms before running the analysis.")
            return

        predictions = [
            {"Model": model_name, "Predicted Disease": predict_disease(model, selected_symptoms)}
            for model_name, model in models.items()
        ]
        results_df = pd.DataFrame(predictions)
        vote_df = (
            results_df["Predicted Disease"]
            .value_counts()
            .rename_axis("Disease")
            .reset_index(name="Votes")
            .set_index("Disease")
        )
        agreement_df = (
            pd.crosstab(results_df["Model"], results_df["Predicted Disease"])
            .astype(int)
        )

        st.subheader("Prediction Results")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        top_prediction = vote_df["Votes"].idxmax()
        top_votes = int(vote_df["Votes"].max())
        st.success(f"Most common prediction: {top_prediction} ({top_votes} model vote{'s' if top_votes != 1 else ''})")

        chart_col1, chart_col2 = st.columns(2, gap="large")
        with chart_col1:
            st.subheader("Disease Vote Distribution")
            st.bar_chart(vote_df)
        with chart_col2:
            st.subheader("Model Agreement Graph")
            st.bar_chart(agreement_df)


if __name__ == "__main__":
    main()

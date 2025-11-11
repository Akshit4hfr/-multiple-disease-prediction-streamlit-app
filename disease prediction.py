# health_assistant_streamlit_.py

import os
import pickle
from datetime import datetime
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Health Assistant — Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
HISTORY_CSV = os.path.join(BASE_DIR, "history.csv")

# -------------------- STYLES --------------------
st.markdown(
    """
    <style>
    .app-title {font-size: 26px; font-weight:700}
    .card {background: #ffffff; padding: 18px; border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.06)}
    .muted {color: #6b7280}
    .small {font-size:0.9rem}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load model {os.path.basename(path)}: {e}")
        return None

# Predict with confidence
def predict_with_confidence(model, X):
    if model is None:
        return None, None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([X])
            conf = float(proba[0][1]) if proba.shape[1] == 2 else float(max(proba[0]))
            label = int(conf >= 0.5)
            return label, conf
        else:
            pred = model.predict([X])[0]
            return int(pred), 1.0
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# -------------------- HISTORY HANDLERS --------------------
if "history" not in st.session_state:
    st.session_state.history = []

def add_history(entry, persist=False):
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:1000]
    if persist:
        try:
            pd.DataFrame(st.session_state.history).to_csv(HISTORY_CSV, index=False)
        except Exception as e:
            st.warning(f"Could not persist history: {e}")

def load_persisted_history():
    if os.path.exists(HISTORY_CSV):
        try:
            df = pd.read_csv(HISTORY_CSV)
            st.session_state.history = df.to_dict(orient="records")
            return True
        except Exception as e:
            st.warning(f"Failed to read persisted history: {e}")
    return False

def history_df():
    if not st.session_state.history:
        return None
    df = pd.DataFrame(st.session_state.history)
    if "input" in df.columns:
        try:
            inputs_df = pd.DataFrame(df["input"].tolist())
            df = pd.concat([df.drop(columns=["input"]), inputs_df], axis=1)
        except Exception:
            pass
    return df

# -------------------- LOAD MODELS --------------------
with st.spinner("Loading models..."):
    diabetes_model = load_model(os.path.join(MODELS_DIR, "diabetes_model.sav"))
    heart_model = load_model(os.path.join(MODELS_DIR, "heart_disease_model.sav"))
    parkinsons_model = load_model(os.path.join(MODELS_DIR, "parkinsons_model.sav"))
    liver_model = load_model(os.path.join(MODELS_DIR, "liver_disease_model.sav"))
    kidney_model = load_model(os.path.join(MODELS_DIR, "kidney_disease_model.sav"))

models_available = {
    "Diabetes": diabetes_model is not None,
    "Heart Disease": heart_model is not None,
    "Parkinson's": parkinsons_model is not None,
    "Liver Disease": liver_model is not None,
    "Kidney Disease": kidney_model is not None,
}

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("<div class='app-title'>Health Assistant</div>", unsafe_allow_html=True)
    st.write("Multi-disease quick predictions · interactive charts")

    gender = option_menu(
        "Select Gender",
        ["Male", "Female"],
        icons=["person", "person-fill"],
        menu_icon="gender-ambiguous",
        default_index=0
    )

    page = None
    if gender in ["Male", "Female"]:
        page = option_menu(
            f"{gender} — Diseases",
            ["Dashboard", "Diabetes", "Heart Disease", "Parkinson's", "Liver Disease", "Kidney Disease", "History"],
            icons=["bar-chart-line", "activity", "heart", "person", "clock-history"],
            menu_icon="stethoscope",
            default_index=0
        )

    st.divider()
    st.checkbox("Persist history to disk", key="persist_history")
    if st.button("Load persisted history"):
        if load_persisted_history():
            st.success("Loaded persisted history")

# -------------------- HELPER --------------------
def show_result(title, label, confidence):
    emoji = "✅" if label == 1 else "❌"
    result_text = "Positive" if label == 1 else "Negative"
    pct = f"{confidence*100:.1f}%" if confidence is not None else "N/A"
    st.markdown(f"<div class='card'>### {title}<br><b>{emoji} {result_text}</b> · Confidence: <b>{pct}</b></div>", unsafe_allow_html=True)

# -------------------- DASHBOARD --------------------
if page == "Dashboard":
    st.title("Overview Dashboard")
    df = history_df()
    if df is not None and not df.empty:
        counts = df["disease"].value_counts().reset_index()
        counts.columns = ["Disease", "Count"]
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.pie(counts, names="Disease", values="Count", title="Predictions by Disease")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            avg_conf = df.groupby("disease")["confidence"].mean().reset_index()
            fig2 = px.bar(avg_conf, x="disease", y="confidence", text=avg_conf["confidence"].apply(lambda x: f"{x:.2f}"), title="Average Confidence by Disease")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No predictions yet. Try using one of the disease pages to generate results.")

# -------------------- DIABETES (GENDER-SPECIFIC) --------------------
if page == "Diabetes":
    st.title(f"Diabetes Prediction — {gender}")

    left, right = st.columns([2, 1])

    with left:
        if gender == "Male":
            st.subheader("Male Health Details")
            with st.form("male_diabetes_form"):
                glucose = st.number_input("Glucose (mg/dL)", 0.0, 300.0, 120.0)
                bp = st.number_input("Blood Pressure (mm Hg)", 0.0, 200.0, 70.0)
                skin = st.number_input("Skin Thickness (mm)", 0.0, 100.0, 20.0)
                insulin = st.number_input("Insulin (mu U/ml)", 0.0, 900.0, 80.0)
                bmi = st.number_input("BMI", 0.0, 80.0, 25.0)
                dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.35, format="%.3f")
                age = st.number_input("Age", 0, 120, 30)
                exercise = st.selectbox("Regular Exercise?", ["Yes", "No"])
                alcohol = st.selectbox("Alcohol Consumption?", ["No", "Yes"])
                submitted = st.form_submit_button("Predict")

            if submitted:
                X = [0, glucose, bp, skin, insulin, bmi, dpf, age]
                label, confidence = predict_with_confidence(diabetes_model, X)
                if label is not None:
                    show_result("Diabetes Result (Male)", label, confidence)
                    add_history({
                        "timestamp": datetime.now().isoformat(),
                        "disease": "Diabetes (Male)",
                        "input": X,
                        "prediction": int(label),
                        "confidence": float(confidence)
                    }, persist=st.session_state.get("persist_history", False))

        elif gender == "Female":
            st.subheader("Female Health Details")
            with st.form("female_diabetes_form"):
                pregnancies = st.number_input("Pregnancies", 0, 20, 1)
                glucose = st.number_input("Glucose (mg/dL)", 0.0, 300.0, 120.0)
                bp = st.number_input("Blood Pressure (mm Hg)", 0.0, 200.0, 70.0)
                skin = st.number_input("Skin Thickness (mm)", 0.0, 100.0, 20.0)
                insulin = st.number_input("Insulin (mu U/ml)", 0.0, 900.0, 80.0)
                bmi = st.number_input("BMI", 0.0, 80.0, 25.0)
                dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.35, format="%.3f")
                age = st.number_input("Age", 0, 120, 30)
                menopause = st.selectbox("Post-menopausal?", ["No", "Yes"])
                submitted = st.form_submit_button("Predict")

            if submitted:
                X = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
                label, confidence = predict_with_confidence(diabetes_model, X)
                if label is not None:
                    show_result("Diabetes Result (Female)", label, confidence)
                    add_history({
                        "timestamp": datetime.now().isoformat(),
                        "disease": "Diabetes (Female)",
                        "input": X,
                        "prediction": int(label),
                        "confidence": float(confidence)
                    }, persist=st.session_state.get("persist_history", False))

    with right:
        st.markdown("### Information")
        if gender == "Male":
            st.info("Common male diabetes risk factors: high BMI, alcohol use, stress, lack of exercise.")
        else:
            st.info("Common female diabetes risk factors: pregnancy, menopause, hormonal changes.")

        # Add visualization of recent diabetes predictions
        df = history_df()
        if df is not None and not df.empty:
            diabetes_df = df[df["disease"].str.contains("Diabetes", na=False)]
            if not diabetes_df.empty:
                st.markdown("### Diabetes Trends")
                try:
                    fig = px.line(diabetes_df, x="timestamp", y="confidence", color="disease", title="Confidence over Time")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not draw chart: {e}")


# -------------------- LIVER DISEASE --------------------
if page == "Liver Disease":
    st.title(f"Liver Disease Prediction — {gender}")

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Liver Health Parameters")
        with st.form("liver_form"):
            age_ld = st.number_input("Age", 1, 120, 40)
            total_bil = st.number_input("Total Bilirubin", 0.0, 50.0, 1.5, format="%.2f")
            direct_bil = st.number_input("Direct Bilirubin", 0.0, 50.0, 0.5, format="%.2f")
            alk_phos = st.number_input("Alkaline Phosphotase (U/L)", 0, 2000, 200)
            alt = st.number_input("Alanine Aminotransferase (ALT)", 0, 2000, 60)
            ast = st.number_input("Aspartate Aminotransferase (AST)", 0, 2000, 70)
            total_prot = st.number_input("Total Proteins (g/dL)", 0.0, 12.0, 6.5, format="%.2f")
            albumin = st.number_input("Albumin (g/dL)", 0.0, 10.0, 3.5, format="%.2f")
            agr = st.number_input("Albumin and Globulin Ratio", 0.0, 5.0, 1.1, format="%.2f")
            submitted_ld = st.form_submit_button("Predict")

        if submitted_ld:
            g = 1 if gender == "Male" else 0
            X = [age_ld, g, total_bil, direct_bil, alk_phos, alt, ast, total_prot, albumin, agr]
            label, confidence = predict_with_confidence(liver_model, X)
            if label is not None:
                show_result("Liver Disease Result", label, confidence)
                add_history({
                    "timestamp": datetime.now().isoformat(),
                    "disease": "Liver Disease",
                    "input": X,
                    "prediction": int(label),
                    "confidence": float(confidence)
                }, persist=st.session_state.get("persist_history", False))

    with right:
        st.markdown("### Information")
        st.info("Liver disease risk is associated with abnormal liver enzymes, high bilirubin levels, and low albumin. Consult a doctor for confirmation.")

        df = history_df()
        if df is not None and not df.empty:
            liver_df = df[df["disease"] == "Liver Disease"]
            if not liver_df.empty:
                st.markdown("### Liver Disease Confidence Trends")
                try:
                    fig = px.line(liver_df, x="timestamp", y="confidence", title="Liver Disease Confidence Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not draw chart: {e}")


# -------------------- KIDNEY DISEASE --------------------
if page == "Kidney Disease":
    st.title("Kidney Disease Prediction")

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Kidney Health Parameters")
        with st.form("kidney_form"):
            age = st.number_input("Age", 1, 120, 40)
            bp = st.number_input("Blood Pressure (mmHg)", 0.0, 200.0, 80.0, format="%.1f")
            sg = st.number_input("Specific Gravity", 1.000, 1.040, 1.015, format="%.3f")
            al = st.number_input("Albumin Level", 0, 5, 1)
            su = st.number_input("Sugar Level", 0, 5, 0)
            bgr = st.number_input("Blood Glucose Random (mg/dL)", 0.0, 500.0, 120.0, format="%.1f")
            bu = st.number_input("Blood Urea (mg/dL)", 0.0, 300.0, 40.0, format="%.1f")
            sc = st.number_input("Serum Creatinine (mg/dL)", 0.0, 15.0, 1.2, format="%.2f")
            sod = st.number_input("Sodium (mEq/L)", 0.0, 200.0, 135.0, format="%.1f")
            pot = st.number_input("Potassium (mEq/L)", 0.0, 10.0, 4.5, format="%.1f")
            hemo = st.number_input("Hemoglobin (g/dL)", 0.0, 25.0, 13.0, format="%.1f")
            pcv = st.number_input("Packed Cell Volume", 0.0, 70.0, 40.0, format="%.1f")
            wc = st.number_input("White Blood Cell Count (cells/cmm)", 0.0, 25000.0, 8000.0, format="%.1f")
            rc = st.number_input("Red Blood Cell Count (millions/cmm)", 0.0, 10.0, 5.0, format="%.2f")
            submitted_kd = st.form_submit_button("Predict")

        if submitted_kd:
            X = [age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc]
            label, confidence = predict_with_confidence(kidney_model, X)
            if label is not None:
                show_result("Kidney Disease Result", label, confidence)
                add_history({
                    "timestamp": datetime.now().isoformat(),
                    "disease": "Kidney Disease",
                    "input": X,
                    "prediction": int(label),
                    "confidence": float(confidence)
                }, persist=st.session_state.get("persist_history", False))

    with right:
        st.markdown("### Information")
        st.info("Kidney disease risk is linked to abnormal urea, creatinine, and electrolyte levels. "
                "This tool provides an early prediction; always consult a doctor for professional evaluation.")

        df = history_df()
        if df is not None and not df.empty:
            kidney_df = df[df["disease"] == "Kidney Disease"]
            if not kidney_df.empty:
                st.markdown("### Kidney Disease Confidence Trends")
                try:
                    fig = px.line(kidney_df, x="timestamp", y="confidence", title="Kidney Disease Confidence Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not draw chart: {e}")


# -------------------- HEART DISEASE --------------------
if page == "Heart Disease":
    st.title(f"Heart Disease Prediction — {gender}")

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Heart Health Parameters")
        with st.form("heart_form"):
            age_h = st.number_input("Age", 1, 120, 55)
            sex_h = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type (0: typical angina ... 3: asymptomatic)", [0,1,2,3], index=0)
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 0, 300, 130)
            chol = st.number_input("Serum Cholesterol (mg/dl)", 0, 600, 250)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No","Yes"])
            restecg = st.selectbox("Resting ECG results (0,1,2)", [0,1,2], index=0)
            thalach = st.number_input("Max Heart Rate Achieved", 0, 300, 150)
            exang = st.selectbox("Exercise Induced Angina?", ["No","Yes"])
            oldpeak = st.number_input("ST depression induced by exercise", 0.0, 10.0, 1.0, format="%.2f")
            slope = st.selectbox("Slope of peak exercise ST segment (0,1,2)", [0,1,2], index=1)
            ca = st.selectbox("Number of major vessels (0-4) colored by fluoroscopy", [0,1,2,3,4], index=0)
            thal = st.selectbox("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)", [1,2,3], index=1)
            submitted_h = st.form_submit_button("Predict")

        if submitted_h:
            sex_val = 1 if sex_h == "Male" else 0
            fbs_val = 1 if fbs == "Yes" else 0
            exang_val = 1 if exang == "Yes" else 0
            X = [age_h, sex_val, cp, trestbps, chol, fbs_val, restecg, thalach, exang_val, oldpeak, slope, ca, thal]
            label, confidence = predict_with_confidence(heart_model, X)
            if label is not None:
                show_result("Heart Disease Result", label, confidence)
                add_history({
                    "timestamp": datetime.now().isoformat(),
                    "disease": "Heart Disease",
                    "input": X,
                    "prediction": int(label),
                    "confidence": float(confidence)
                }, persist=st.session_state.get("persist_history", False))

    with right:
        st.markdown("### Information")
        st.info("Heart disease risk factors include high BP, high cholesterol, and exercise-induced angina. This is a screening prediction; consult a cardiologist for diagnosis.")

        df = history_df()
        if df is not None and not df.empty:
            heart_df = df[df["disease"] == "Heart Disease"]
            if not heart_df.empty:
                st.markdown("### Heart Disease Confidence Trends")
                try:
                    fig = px.line(heart_df, x="timestamp", y="confidence", title="Heart Disease Confidence Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not draw chart: {e}")


# -------------------- PARKINSON'S DISEASE --------------------
if page == "Parkinson's":
    st.title(f"Parkinson's Disease Prediction — {gender}")

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Voice / Motor Features (approximate values)")
        with st.form("parkinsons_form"):
            MDVP_Fo = st.number_input("MDVP:Fo(Hz)", 0.0, 500.0, 119.992, format="%.3f")
            MDVP_Fhi = st.number_input("MDVP:Fhi(Hz)", 0.0, 1000.0, 157.302, format="%.3f")
            MDVP_Flo = st.number_input("MDVP:Flo(Hz)", 0.0, 500.0, 74.997, format="%.3f")
            MDVP_Jitter = st.number_input("MDVP:Jitter(%)", 0.0, 1.0, 0.00784, format="%.5f")
            MDVP_Shimmer = st.number_input("MDVP:Shimmer", 0.0, 1.0, 0.04374, format="%.5f")
            NHR = st.number_input("NHR", 0.0, 1.0, 0.02211, format="%.5f")
            HNR = st.number_input("HNR", 0.0, 100.0, 21.033, format="%.3f")
            RPDE = st.number_input("RPDE", 0.0, 1.0, 0.414783, format="%.6f")
            DFA = st.number_input("DFA", 0.0, 2.0, 0.815285, format="%.6f")
            spread1 = st.number_input("spread1", -5.0, 5.0, -4.813031, format="%.6f")
            spread2 = st.number_input("spread2", -10.0, 10.0, 2.057621, format="%.6f")
            D2 = st.number_input("D2", 0.0, 10.0, 2.302618, format="%.6f")
            PPE = st.number_input("PPE", 0.0, 5.0, 0.284654, format="%.6f")
            submitted_p = st.form_submit_button("Predict")

        if submitted_p:
            X = [MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_Shimmer, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
            label, confidence = predict_with_confidence(parkinsons_model, X)
            if label is not None:
                show_result("Parkinson's Result", label, confidence)
                add_history({
                    "timestamp": datetime.now().isoformat(),
                    "disease": "Parkinson's",
                    "input": X,
                    "prediction": int(label),
                    "confidence": float(confidence)
                }, persist=st.session_state.get("persist_history", False))

    with right:
        st.markdown("### Information")
        st.info("Parkinson's disease affects motor control and speech. Voice-based features can help screen for Parkinson's, but this is not a diagnosis. Consult a neurologist for clinical evaluation.")

        df = history_df()
        if df is not None and not df.empty:
            parkinsons_df = df[df["disease"] == "Parkinson's"]
            if not parkinsons_df.empty:
                st.markdown("### Parkinson's Confidence Trends")
                try:
                    fig = px.line(parkinsons_df, x="timestamp", y="confidence", title="Parkinson's Confidence Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not draw chart: {e}")

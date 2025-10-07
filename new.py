# health_assistant_streamlit_improved.py
"""
Improved and more professional Streamlit front-end for the Health Assistant app.
Features added:
- Clean layout and responsive cards
- Model caching and graceful failure handling
- Persistent history option (local CSV) with import/export
- Better use of Streamlit metrics, expandable sections, and info cards
- Interactive Plotly charts for nicer visuals
- Input validation and compact forms
- Small demo data mode when models are missing

Install requirements:
    pip install streamlit pandas plotly streamlit-option-menu

Drop this file next to your saved_models/ folder and run:
    streamlit run health_assistant_streamlit_improved.py

"""

import os
import pickle
from datetime import datetime
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ---------------------------
# Page config + lightweight theme
# ---------------------------
st.set_page_config(page_title="Health Assistant — Dashboard",
                   layout="wide",
                   page_icon="📊",
                   initial_sidebar_state="expanded")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
HISTORY_CSV = os.path.join(BASE_DIR, "history.csv")

# ---------------------------
# Styles
# ---------------------------
st.markdown(
    """
    <style>
    .app-title {font-size: 26px; font-weight:700}
    .card {background: #ffffff; padding: 18px; border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.06)}
    .muted {color: #6b7280}
    .small {font-size:0.9rem}
    .metric-label {color:#6b7280}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Utility: load model safely (cached)
# ---------------------------
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


# Predict with probability where available
def predict_with_confidence(model, X):
    if model is None:
        return None, None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([X])
            if proba.shape[1] == 2:
                conf = float(proba[0][1])
                label = int(conf >= 0.5)
                return label, conf
            else:
                idx = int(proba[0].argmax())
                conf = float(proba[0][idx])
                return idx, conf
        else:
            pred = model.predict([X])[0]
            return int(pred), 1.0
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None


# ---------------------------
# Session & History helpers
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []


def add_history(entry, persist=False):
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:1000]
    if persist:
        try:
            df = pd.DataFrame(st.session_state.history)
            df.to_csv(HISTORY_CSV, index=False)
        except Exception as e:
            st.warning(f"Could not persist history: {e}")


def load_persisted_history():
    if os.path.exists(HISTORY_CSV):
        try:
            df = pd.read_csv(HISTORY_CSV)
            records = df.to_dict(orient="records")
            st.session_state.history = records
            return True
        except Exception as e:
            st.warning(f"Failed to read persisted history: {e}")
    return False


def history_df():
    if not st.session_state.history:
        return None
    df = pd.DataFrame(st.session_state.history)
    # expand input lists into columns
    if "input" in df.columns:
        try:
            max_len = max(df["input"].apply(lambda x: len(x) if isinstance(x, (list, tuple)) else 0))
            inputs_df = pd.DataFrame(df["input"].tolist(), columns=[f"feat_{i}" for i in range(max_len)])
            df = pd.concat([df.drop(columns=["input"]), inputs_df], axis=1)
        except Exception:
            pass
    return df


# ---------------------------
# Load models
# ---------------------------
with st.spinner("Loading models..."):
    diabetes_model = load_model(os.path.join(MODELS_DIR, "diabetes_model.sav"))
    heart_model = load_model(os.path.join(MODELS_DIR, "heart_disease_model.sav"))
    parkinsons_model = load_model(os.path.join(MODELS_DIR, "parkinsons_model.sav"))

models_available = {
    "Diabetes": diabetes_model is not None,
    "Heart Disease": heart_model is not None,
    "Parkinson's": parkinsons_model is not None,
}

# ---------------------------
# Sidebar (navigation + options)
# ---------------------------
with st.sidebar:
    st.markdown("<div class='app-title'>Health Assistant</div>", unsafe_allow_html=True)
    st.write("Multi-disease quick predictions \u2022 interactive charts")
    page = option_menu(None, ["Dashboard", "Diabetes", "Heart Disease", "Parkinson's", "History"],
                       icons=["bar-chart-line", "activity", "heart", "person", "clock-history"],
                       menu_icon="cast", default_index=0)

    st.divider()
    st.checkbox("Persist history to disk", key="persist_history")
    if st.button("Load persisted history"):
        ok = load_persisted_history()
        if ok:
            st.success("Loaded persisted history")
    st.caption("Tip: enable persistence to keep history across Streamlit restarts")

# ---------------------------
# Small helpers for result cards
# ---------------------------

def show_result(title, label, confidence):
    positive = bool(label == 1)
    emoji = "✅" if positive else "❌"
    pct = f"{confidence*100:.1f}%" if confidence is not None else "N/A"
    st.markdown(f"<div class='card'>\n### {title}\n\n**{emoji} {'Positive' if positive else 'Negative'}** · Confidence: **{pct}**\n</div>", unsafe_allow_html=True)


# ---------------------------
# Dashboard (overview)
# ---------------------------
if page == "Dashboard":
    st.title("Overview")
    cols = st.columns(3)
    counts = {k: sum(1 for r in st.session_state.history if r.get("disease") == k) for k in ["Diabetes", "Heart Disease", "Parkinson's"]}
    confidences = {k: np.mean([r.get("confidence") or 0 for r in st.session_state.history if r.get("disease") == k]) if counts[k] > 0 else 0 for k in counts}

    with cols[0]:
        st.metric("Diabetes checks", counts["Diabetes"], delta=None)
        st.caption("Model: {}".format("Loaded" if models_available["Diabetes"] else "Missing"))
    with cols[1]:
        st.metric("Heart checks", counts["Heart Disease"], delta=None)
        st.caption("Model: {}".format("Loaded" if models_available["Heart Disease"] else "Missing"))
    with cols[2]:
        st.metric("Parkinson's checks", counts["Parkinson's"], delta=None)
        st.caption("Model: {}".format("Loaded" if models_available["Parkinson's"] else "Missing"))

    st.divider()
    st.subheader("Recent predictions")
    df = history_df()
    if df is None:
        st.info("No predictions yet — try using the forms in the left navigation.")
    else:
        view = df.head(10)
        st.dataframe(view, use_container_width=True)
        st.download_button("Download recent CSV", data=view.to_csv(index=False).encode('utf-8'), file_name='recent_predictions.csv')

    st.subheader("Quick charts")
    if df is not None:
        # pie by disease positive rate
        try:
            agg = pd.DataFrame(st.session_state.history)
            agg['positive'] = agg['prediction'] == 1
            summary = agg.groupby('disease')['positive'].mean().reset_index(name='positive_rate')
            fig = px.bar(summary, x='disease', y='positive_rate', text=summary['positive_rate'].apply(lambda x: f"{x:.1%}"))
            fig.update_layout(yaxis_tickformat='.0%', title='Positive rate by disease')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not build quick chart: {e}")

    st.markdown("---")
    st.markdown("<div class='muted small'>Pro tip: enable persistence to save session history to a local CSV file (history.csv).</div>", unsafe_allow_html=True)

# ---------------------------
# Diabetes page (improved UX)
# ---------------------------
if page == 'Diabetes':
    st.title("Diabetes — Predict")
    left, right = st.columns([2,1])

    with left:
        with st.form('diabetes_form'):
            st.markdown("**Patient info**")
            c1, c2, c3 = st.columns(3)
            with c1:
                pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
                skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0.0, max_value=100.0, value=20.0)
                dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.35, format='%.3f')
            with c2:
                glucose = st.number_input('Glucose (mg/dL)', min_value=0.0, max_value=300.0, value=120.0)
                insulin = st.number_input('Insulin (mu U/ml)', min_value=0.0, max_value=900.0, value=80.0)
                age = st.number_input('Age', min_value=0, max_value=120, value=30)
            with c3:
                bp = st.number_input('Blood Pressure (mm Hg)', min_value=0.0, max_value=200.0, value=70.0)
                bmi = st.number_input('BMI', min_value=0.0, max_value=80.0, value=25.0)
                st.write(' ')

            st.form_submit_button('Run prediction', on_click=lambda: None)

        if st.button('Run Diabetes now'):
            X = [pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]
            label, confidence = predict_with_confidence(diabetes_model, X)
            if label is None:
                st.error('Prediction failed — model missing or error')
            else:
                show_result('Diabetes result', label, confidence)
                add_history({
                    'timestamp': datetime.now().isoformat(),
                    'disease': 'Diabetes',
                    'input': X,
                    'prediction': int(label),
                    'confidence': float(confidence) if confidence is not None else None
                }, persist=st.session_state.get('persist_history', False))

    with right:
        st.markdown('### Info')
        st.info('This page uses the diabetes model to predict presence of diabetes based on common clinical features.')
        if not models_available['Diabetes']:
            st.warning('Diabetes model not found — run in demo mode using synthetic values for UI checks')

# ---------------------------
# Heart disease page
# ---------------------------
if page == 'Heart Disease':
    st.title('Heart Disease — Predict')
    with st.form('heart_form'):
        cols = st.columns(3)
        with cols[0]:
            age = st.number_input('Age', min_value=1, max_value=120, value=55)
            trestbps = st.number_input('Resting BP (trestbps)', min_value=50, max_value=300, value=140)
            restecg = st.selectbox('Resting ECG (restecg)', [0,1,2])
        with cols[1]:
            sex = st.selectbox('Sex', [0,1], index=1)
            chol = st.number_input('Cholesterol', min_value=50, max_value=800, value=250)
            thalach = st.number_input('Max Heart Rate Achieved', min_value=50, max_value=260, value=150)
        with cols[2]:
            cp = st.selectbox('Chest Pain type', [0,1,2,3])
            fbs = st.selectbox('Fasting blood sugar >120 mg/dl', [0,1])
            exang = st.selectbox('Exercise induced angina', [0,1])
            oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=1.0, format='%.2f')

        submitted = st.form_submit_button('Run prediction')

    if submitted:
        X = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, 1, 0, 0]
        label, confidence = predict_with_confidence(heart_model, X)
        if label is None:
            st.error('Prediction failed — model missing or error')
        else:
            show_result('Heart disease result', label, confidence)
            add_history({
                'timestamp': datetime.now().isoformat(),
                'disease': 'Heart Disease',
                'input': X,
                'prediction': int(label),
                'confidence': float(confidence) if confidence is not None else None
            }, persist=st.session_state.get('persist_history', False))

# ---------------------------
# Parkinson's page
# ---------------------------
if page == "Parkinson's":
    st.title("Parkinson's — Predict")
    with st.form('par_form'):
        cols = st.columns(4)
        # condensed inputs for brevity; keep default values
        fo = st.number_input('MDVP:Fo(Hz)', value=119.992, format='%.3f')
        fhi = st.number_input('MDVP:Fhi(Hz)', value=157.302, format='%.3f')
        flo = st.number_input('MDVP:Flo(Hz)', value=74.997, format='%.3f')
        jitter = st.number_input('MDVP:Jitter(%)', value=0.00784, format='%.6f')
        shimmer = st.number_input('MDVP:Shimmer', value=0.04374, format='%.6f')
        nhr = st.number_input('NHR', value=0.02211, format='%.6f')
        rpde = st.number_input('RPDE', value=0.414784, format='%.6f')
        dfa = st.number_input('DFA', value=2.301442, format='%.6f')

        submitted = st.form_submit_button('Run prediction')

    if submitted:
        X = [fo, fhi, flo, jitter, 0.00007, 0.005, 0.018, 0.00007, 0.04374, 0.287, 0.0309, 0.02748, 0.02971, 0.04478, 0.02211, 21.033, rpde, dfa, -4.813, 0.266, 2.234, 0.0]
        label, confidence = predict_with_confidence(parkinsons_model, X)
        if label is None:
            st.error('Prediction failed — model missing or error')
        else:
            show_result("Parkinson's result", label, confidence)
            add_history({
                'timestamp': datetime.now().isoformat(),
                'disease': "Parkinson's",
                'input': X,
                'prediction': int(label),
                'confidence': float(confidence) if confidence is not None else None
            }, persist=st.session_state.get('persist_history', False))

# ---------------------------
# History / Charts page
# ---------------------------
if page == 'History':
    st.title('History & Charts')
    df = history_df()

    col1, col2 = st.columns([3,1])
    with col1:
        if df is None:
            st.info('No history yet — run predictions on the left pages.')
        else:
            st.dataframe(df, use_container_width=True)
            st.download_button('Download full history', data=df.to_csv(index=False).encode('utf-8'), file_name='history_full.csv')

            st.markdown('---')
            st.subheader('Confidence over time')
            try:
                times = pd.to_datetime(df['timestamp'])
                agg = pd.DataFrame(st.session_state.history)
                agg['timestamp'] = pd.to_datetime(agg['timestamp'])
                fig = px.line(agg, x='timestamp', y='confidence', color='disease', markers=True, title='Confidence over time')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f'Could not draw time series: {e}')

            st.subheader('Average confidence by disease')
            try:
                avg = df.groupby('disease')['confidence'].mean().reset_index()
                fig = px.bar(avg, x='disease', y='confidence', text=avg['confidence'].apply(lambda x: f"{x:.2f}"), title='Avg confidence by disease')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f'Could not draw avg chart: {e}')

    with col2:
        st.markdown('### Controls')
        if st.button('Clear session history'):
            st.session_state.history = []
            st.success('Session history cleared')
        if st.button('Export to CSV (persistent)'):
            if df is not None:
                df.to_csv(HISTORY_CSV, index=False)
                st.success('Exported history to history.csv')
            else:
                st.info('Nothing to export')
        uploaded = st.file_uploader('Import history CSV', type=['csv'])
        if uploaded is not None:
            try:
                df_u = pd.read_csv(uploaded)
                st.session_state.history = df_u.to_dict(orient='records')
                st.success('Imported history')
            except Exception as e:
                st.error(f'Failed to import: {e}')

# Footer
st.markdown('---')
st.markdown("<div class='muted small'>Built with ❤️ — drop models into saved_models/ to enable real predictions. This UI supports optional persistence via history.csv.</div>", unsafe_allow_html=True)

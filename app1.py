import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️")


@st.cache_resource
def load_artifacts():
    # Change these file names if your saved SVM files are named differently
    model = joblib.load("logist_heartdis.pkl")
    scaler = joblib.load("scalerlo.pkl")
    cols = joblib.load("coloflo.pkl")
    if isinstance(cols[0], list):
        cols = cols[0]
    return model, scaler, list(cols)


model, scaler, cols = load_artifacts()

st.title("HEART DISEASE RISK PREDICTION")
st.markdown("Provide the following details:")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest = st.selectbox("Chest pain type", ["ATA", "NAP", "TA", "ASY"])
rest_bp = st.number_input("Resting blood pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Cholesterol (mg/dL)", 0, 600, 200)
fasting = st.selectbox("Fasting blood sugar > 120 mg/dL (1 = yes, 0 = no)", [0, 1])
ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max heart rate", 60, 220, 150)
angina = st.selectbox("Exercise-induced angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, 0.1)
slope = st.selectbox("ST slope", ["Up", "Flat", "Down"])

if st.button("PREDICT"):
    # Column names follow the Kaggle dataset after one-hot encoding.
    # Every category is set, so it works whether or not drop_first=True was used;
    # reindex() below keeps only the columns the model was trained on.
    raw_input = {
        "Age": age,
        "RestingBP": rest_bp,
        "Cholesterol": chol,
        "FastingBS": fasting,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": int(sex == "M"),
        "Sex_F": int(sex == "F"),
        "ChestPainType_ATA": int(chest == "ATA"),
        "ChestPainType_NAP": int(chest == "NAP"),
        "ChestPainType_TA": int(chest == "TA"),
        "ChestPainType_ASY": int(chest == "ASY"),
        "RestingECG_Normal": int(ecg == "Normal"),
        "RestingECG_ST": int(ecg == "ST"),
        "RestingECG_LVH": int(ecg == "LVH"),
        "ExerciseAngina_Y": int(angina == "Y"),
        "ExerciseAngina_N": int(angina == "N"),
        "ST_Slope_Up": int(slope == "Up"),
        "ST_Slope_Flat": int(slope == "Flat"),
        "ST_Slope_Down": int(slope == "Down"),
    }

    missing = [c for c in cols if c not in raw_input]
    if missing:
        st.warning(f"These training columns were not filled by the app: {missing}")

    input_df = pd.DataFrame([raw_input]).reindex(columns=cols, fill_value=0)
    scaled = scaler.transform(input_df)

    prediction = model.predict(scaled)[0]
    if prediction == 1:
        st.error("HIGH RISK OF HEART DISEASE")
    else:
        st.success("LOW RISK OF HEART DISEASE")

    if hasattr(model, "predict_proba"):
        st.write(f"Estimated risk probability: {model.predict_proba(scaled)[0][1]:.1%}")

    st.caption("This tool is for educational purposes only and is not a medical diagnosis.")

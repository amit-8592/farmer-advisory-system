import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# PAGE SETUP
# -------------------------------
st.set_page_config(page_title="Farmer Scheme Predictor", layout="centered")
st.title("🌾 Farmer Scheme Recommendation System")

# -------------------------------
# LOAD MODEL (NO PATH NEEDED)
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    le = joblib.load("label_encoder.pkl")
    columns = joblib.load("columns.pkl")
    return model, le, columns

with st.spinner("Loading model... ⏳"):
    model, le, columns = load_model()

st.success("Model loaded successfully ✅")

# -------------------------------
# USER INPUT
# -------------------------------
land = st.selectbox("Land Size", ["Small", "Medium", "Large"])
income = st.selectbox("Income Level", ["Low", "Medium", "High"])
crop = st.selectbox("Crop", ["Wheat", "Rice", "Maize", "Cotton", "Soybean"])
irrigation = st.selectbox("Irrigation", ["Yes", "No"])
soil = st.selectbox("Soil Type", ["Loamy", "Clay", "Sandy", "Black"])
loan = st.selectbox("Loan Status", ["Yes", "No"])
weather = st.selectbox("Weather Risk", ["Low", "Medium", "High"])
exp = st.selectbox("Farm Experience", ["Low", "Medium", "High"])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Scheme"):

    # Mapping
    income_map = {"Low": 1, "Medium": 2, "High": 3}
    land_map = {"Small": 1, "Medium": 2, "Large": 3}
    exp_map = {"Low": 1, "Medium": 2, "High": 3}
    loan_map = {"No": 0, "Yes": 1}

    land_num = land_map[land]
    income_num = income_map[income]
    exp_num = exp_map[exp]
    loan_num = loan_map[loan]

    # Feature Engineering
    risk_score = (
        int(irrigation == "No") +
        int(weather == "High") +
        int(soil == "Sandy")
    )

    input_data = pd.DataFrame([{
        "Land_Size": land_num,
        "Income_Level": income_num,
        "Crop": crop,
        "Irrigation": irrigation,
        "Soil_Type": soil,
        "Loan_Status": loan_num,
        "Weather_Risk": weather,
        "Farm_Experience": exp_num,
        "Land_Income": land_num * income_num,
        "Smart_Factor": exp_num * income_num,
        "Risk_Score": risk_score,
        "Crop_Soil": f"{crop}_{soil}",
        "Financial_Score": income_num + loan_num,
        "Eligible_PM_Kisan": int(land_num <= 2),
        "Eligible_Loan_Scheme": int(loan_num == 1 and income_num <= 2),
    }])

    # Encoding
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_data)
    probs = model.predict_proba(input_data)

    scheme = le.inverse_transform(prediction)[0]

    # Top-3
    top3 = np.argsort(probs, axis=1)[:, -3:]
    top3_labels = le.inverse_transform(top3[0])

    # Output
    st.success(f"✅ Recommended Scheme: **{scheme}**")

    st.subheader("🔥 Top 3 Recommendations:")
    for s in reversed(top3_labels):
        st.write("👉", s)
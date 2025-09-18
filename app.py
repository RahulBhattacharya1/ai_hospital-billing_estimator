# app.py
import json
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, datetime

st.set_page_config(page_title="Hospital Billing Amount Estimator", layout="wide")
st.title("Hospital Billing Amount Estimator")

# ------------------------------------------------------------
# Load model artifacts
# ------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """
    Expects:
      models/billing_model.joblib
      models/feature_columns.json
    """
    try:
        artifacts = joblib.load("models/billing_model.joblib")
    except FileNotFoundError:
        st.stop()
    try:
        with open("models/feature_columns.json", "r") as f:
            feature_columns = json.load(f)
    except FileNotFoundError:
        st.stop()

    model = artifacts["model"]
    raw_features = artifacts.get("feature_cols_raw", [])
    return model, feature_columns, raw_features

model, FEATURE_COLUMNS, RAW_FEATURES = load_artifacts()

st.markdown(
    "Use the form below for a single estimate or upload a CSV in the sidebar for batch predictions. "
    "The app aligns your inputs to the model’s one-hot schema automatically."
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def to_numeric_safe(val_series):
    """Extract numeric values from mixed strings like '110 mg/dL'."""
    s = pd.Series(val_series)
    num = pd.to_numeric(
        s.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce"
    )
    return num

def compute_los(admit_dt, discharge_dt):
    """Length of stay in days from two date objects; returns 0.0 if invalid."""
    try:
        if isinstance(admit_dt, date):
            admit_dt = pd.Timestamp(admit_dt)
        if isinstance(discharge_dt, date):
            discharge_dt = pd.Timestamp(discharge_dt)
        if pd.isna(admit_dt) or pd.isna(discharge_dt):
            return 0.0
        days = (discharge_dt - admit_dt).days
        return float(max(days, 0))
    except Exception:
        return 0.0

def to_model_frame(df_like: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    One-hot encode input frame and align to training schema.
    Missing columns are added with 0.0; extra columns are ignored.
    """
    X = df_like.copy()

    # Ensure object columns are strings
    for c in X.columns:
        if X[c].dtype == "O":
            X[c] = X[c].astype(str).fillna("")

    dummies = pd.get_dummies(X, drop_first=False, dtype=float)

    # Add missing expected columns
    for col in feature_columns:
        if col not in dummies.columns:
            dummies[col] = 0.0

    # Keep only the columns the model expects, in order
    dummies = dummies[feature_columns]
    return dummies

def template_csv_bytes():
    """Generate a small CSV template users can fill for batch predictions."""
    cols = [
        "Age","Gender","Blood Type","Medical Condition",
        "Hospital","Insurance Provider","Admission Type",
        "Medication","Test Results","Date of Admission","Discharge Date"
    ]
    sample = pd.DataFrame(
        [
            [42,"Male","O+","Diabetes","General Hospital","ACME Health","Emergency","Metformin","110","2025-01-01","2025-01-05"],
            [35,"Female","A+","Hypertension","City Hospital","PrimeCare","Elective","Lisinopril","132","2025-02-10","2025-02-13"]
        ],
        columns=cols
    )
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# ------------------------------------------------------------
# Sidebar: Batch predictions
# ------------------------------------------------------------
st.sidebar.header("Batch Prediction")
st.sidebar.caption("Upload a CSV. You can download a template first.")
st.sidebar.download_button(
    label="Download CSV template",
    data=template_csv_bytes(),
    file_name="billing_batch_template.csv",
    mime="text/csv"
)

batch_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if st.sidebar.button("Predict Batch"):
    if batch_file is None:
        st.sidebar.error("Please upload a CSV file.")
    else:
        try:
            bdf = pd.read_csv(batch_file)
        except Exception as e:
            st.sidebar.error(f"Could not read CSV: {e}")
            st.stop()

        # Optional date parsing and LOS computation
        if {"Date of Admission", "Discharge Date"}.issubset(set(bdf.columns)):
            bdf["Date of Admission"] = pd.to_datetime(bdf["Date of Admission"], errors="coerce")
            bdf["Discharge Date"] = pd.to_datetime(bdf["Discharge Date"], errors="coerce")
            bdf["length_of_stay"] = (bdf["Discharge Date"] - bdf["Date of Admission"]).dt.days
            bdf["length_of_stay"] = bdf["length_of_stay"].clip(lower=0).fillna(0)

        # Build the raw feature frame using whatever of RAW_FEATURES are present
        use_cols = [c for c in RAW_FEATURES if c in bdf.columns]
        if "Test Results" in use_cols:
            bdf["Test Results"] = to_numeric_safe(bdf["Test Results"])
        if "Age" in use_cols:
            bdf["Age"] = pd.to_numeric(bdf["Age"], errors="coerce")

        if not use_cols:
            st.error("Uploaded CSV does not include any of the model’s expected features.")
        else:
            Xm = to_model_frame(bdf[use_cols], FEATURE_COLUMNS)
            preds = model.predict(Xm)
            out = bdf.copy()
            out["Predicted Billing Amount"] = np.round(preds, 2)
            st.success(f"Predicted {len(out)} rows.")
            st.dataframe(out.head(50))
            st.download_button(
                label="Download predictions",
                data=out.to_csv(index=False),
                file_name="billing_predictions.csv",
                mime="text/csv"
            )

# ------------------------------------------------------------
# Single prediction UI
# ------------------------------------------------------------
st.subheader("Single Prediction")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=40)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    blood = st.selectbox("Blood Type", ["A+","A-","B+","B-","AB+","AB-","O+","O-"])
with col2:
    condition = st.text_input("Medical Condition", "Diabetes")
    hospital = st.text_input("Hospital", "General Hospital")
    insurer = st.text_input("Insurance Provider", "ACME Health")
with col3:
    adm_type = st.selectbox("Admission Type", ["Emergency","Elective","Urgent"])
    medication = st.text_input("Medication", "Metformin")
    test_res = st.text_input("Test Results (numeric or text)", "110")

col4, col5 = st.columns(2)
with col4:
    adm_date = st.date_input("Date of Admission", value=date(2025, 1, 1))
with col5:
    dis_date = st.date_input("Discharge Date (optional)", value=date(2025, 1, 5))

los = compute_los(adm_date, dis_date)
st.caption(f"Computed length_of_stay: {los} day(s)")

if st.button("Predict Billing Amount"):
    # Build a single-row raw dictionary using only features known to the model
    row = {
        "Age": age,
        "Gender": gender,
        "Blood Type": blood,
        "Medical Condition": condition,
        "Hospital": hospital,
        "Insurance Provider": insurer,
        "Admission Type": adm_type,
        "Medication": medication,
        "Test Results": to_numeric_safe([test_res]).iloc[0],
        "length_of_stay": los
    }
    row = {k: v for k, v in row.items() if k in RAW_FEATURES}

    Xm = to_model_frame(pd.DataFrame([row]), FEATURE_COLUMNS)
    pred = float(model.predict(Xm)[0])
    st.success(f"Estimated Billing Amount: ${pred:,.2f}")

# ------------------------------------------------------------
# Debug/Info panel
# ------------------------------------------------------------
with st.expander("Show expected model features"):
    st.write("These are the one-hot columns the model expects after encoding:")
    st.write(f"Count: {len(FEATURE_COLUMNS)}")
    st.code("\n".join(FEATURE_COLUMNS[:200]) + ("\n..." if len(FEATURE_COLUMNS) > 200 else ""))

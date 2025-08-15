import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
from utils.prediction_imputation import impute_for_prediction
# --- Setup ---
MODEL_DIR = "models"
REQUIRED_KEY = "columns"

st.set_page_config(page_title="📈 Predict Patron+ Probability", layout="wide")
st.title("📈 Predict Patron+ Probability Using a Trained Model")

# --- Model selection ---
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
model_name = st.selectbox("📂 Select a trained model", sorted(model_files))

# --- File upload ---
uploaded_csv = st.file_uploader("📤 Upload a CSV with customer_no and feature columns (no Donor_Category)", type=["csv"])

if model_name and uploaded_csv:
    try:
        # Load model
        with open(os.path.join(MODEL_DIR, model_name), "rb") as f:
            model_data = joblib.load(f)

        model = model_data["model"]
        required_columns = model_data["columns"]

        # Load uploaded CSV
        df_original = pd.read_csv(uploaded_csv)

        # Check for required columns
        missing_cols = [col for col in required_columns if col not in df_original.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns for this model: {missing_cols}")
        else:
            # Keep customer_no if present
            customer_nos = df_original["customer_no"] if "customer_no" in df_original.columns else None

            # Prepare input for model (drop customer_no if exists)
            X_input = df_original[required_columns].copy()

            # Impute missing values
            X_clean = impute_for_prediction(X_input)

            # Predict Patron+ probability
            probs = model.predict_proba(X_clean)[:, 1]

            # Build result DataFrame
            output_df = df_original.copy()

            # Only show and export the probability column, renamed for clarity
            output_df["Patron+ Alikeness"] = probs
            output_df = output_df[["customer_no", "Patron+ Aliekness"]] if "customer_no" in output_df.columns else output_df[["Patron+ Alikeness"]]

            # Preview
            st.subheader("🔍 Preview of Predictions")
            st.dataframe(output_df.head(20), use_container_width=True)

            # Download
            csv_output = output_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV with Patron+ Probability", csv_output, file_name="patron_probability.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error processing model or CSV: {e}")

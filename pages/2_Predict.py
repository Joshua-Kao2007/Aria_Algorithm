import streamlit as st
import os
import joblib
import pandas as pd
from utils.model_imputation import clean_and_impute
# Constants
MODEL_DIR = "models"

st.set_page_config(page_title="🎯 Predict Patron+ Probability", layout="wide")
st.title("🎯 Predict Patron+ Probability")

# Get all available model files
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]

if not model_files:
    st.warning("⚠️ No models available. Please train a model first.")
    st.stop()

# Select model
selected_model_file = st.selectbox("📂 Choose a trained model", sorted(model_files))
model_path = os.path.join(MODEL_DIR, selected_model_file)

# Load model
try:
    with open(model_path, "rb") as f:
        model_data = joblib.load(f)

    model = model_data["model"]
    required_columns = model_data["columns"]

except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Upload data
st.markdown("📤 Upload a **CSV file** with constituent features (must match training columns):")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        input_df = pd.read_csv(uploaded_file)

        # Drop 'customer_no' if included
        if "customer_no" in input_df.columns:
            customer_ids = input_df["customer_no"]
            input_df = input_df.drop(columns=["customer_no"])
        else:
            customer_ids = None

        # Check for missing columns
        missing_cols = [col for col in required_columns if col not in input_df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.stop()

        # Reorder and filter input columns to match training
        input_df = input_df[required_columns]
        input_df = clean_and_impute(input_df)
        # Predict probabilities
        probs = model.predict_proba(input_df)

        # Find column index for Patron+ (label 1)
        if model.classes_[1] == 1:
            patron_index = 1
        else:
            patron_index = list(model.classes_).index(1)

        input_df["Patron+_Probability"] = probs[:, patron_index]

        # Add back customer_no if it existed
        if customer_ids is not None:
            input_df.insert(0, "customer_no", customer_ids)

        # Show preview
        st.success("✅ Predictions generated!")
        st.dataframe(input_df.head(20), use_container_width=True)

        # Download
        st.download_button(
            label="📥 Download Predictions CSV",
            data=input_df.to_csv(index=False).encode("utf-8"),
            file_name="patron_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error processing CSV: {e}")

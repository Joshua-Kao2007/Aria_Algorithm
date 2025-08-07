import streamlit as st
import os
import joblib
import pandas as pd

st.set_page_config(page_title="View Models", layout="wide")
st.title("📂 View Saved Models")

model_files = [f for f in os.listdir("models") if f.endswith(".pkl")]

selected_model_file = st.selectbox("Select a model to inspect:", model_files)

if selected_model_file:
    model_path = os.path.join("models", selected_model_file)
    model_data = joblib.load(model_path)

    st.markdown(f"### 🧠 Model Name: `{selected_model_file}`")

    # Show model structure
    if "category_names" in model_data and "category_models" in model_data:
        st.subheader("📁 Categories and Models Used")
        for name, model in zip(model_data["category_names"], model_data["category_models"]):
            st.write(f"**{name}** → {model}")

    # Show column names
    st.subheader("🧾 Columns Used")
    st.code(", ".join(model_data["columns"]))

    # Show sample data
    if "X_sample" in model_data:
        st.subheader("🔍 Sample of Data")
        st.dataframe(model_data["X_sample"].head())

        st.subheader("📊 Summary Statistics")
        st.write(model_data["X_sample"].describe().T)

        # If categorical features exist
        cat_cols = model_data["X_sample"].select_dtypes(include="object").columns
        if not cat_cols.empty:
            st.subheader("📈 Mode (Categorical Columns)")
            for col in cat_cols:
                st.write(f"**{col}** → {model_data['X_sample'][col].mode().iloc[0]}")

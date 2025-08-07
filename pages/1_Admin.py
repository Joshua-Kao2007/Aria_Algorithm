import streamlit as st
import os
from utils.model_utils import train_and_save_stacked_model

# -------------------- CONFIG --------------------
ADMIN_PASSWORD = "password"
st.set_page_config(page_title="Admin", layout="centered")
st.title("Admin Panel")

# -------------------- AUTH --------------------
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

if not st.session_state.admin_authenticated:
    password = st.text_input("Enter Admin Password", type="password")
    if st.button("Submit"):
        if password == ADMIN_PASSWORD:
            st.success("Access granted.")
            st.session_state.admin_authenticated = True
        else:
            st.error("Incorrect password.")
    st.stop()

# -------------------- MODEL BUILDER --------------------
st.markdown("---")
st.subheader("Create a New Model")

if "show_builder" not in st.session_state:
    st.session_state.show_builder = False

if st.button("➕ Create New Model"):
    st.session_state.show_builder = True

if st.session_state.show_builder:
    st.markdown("### 📊 Step 1: How many data categories will your model use?")
    num_categories = st.number_input("Enter number of categories:", min_value=1, max_value=10, step=1)

    st.markdown("---")
    st.markdown("### Step 2: Upload & Configure Each Category")

    category_info = []  # Stores (name, file, model)
    model_options = ["Logistic Regression", "Random Forest", "AdaBoost", "KNN", "XGBoost", "Linear Regression", "LightGBM"]

    for i in range(num_categories):
        default_name = f"Category {i+1}"
        name = st.text_input("Category name:", value=default_name, key=f"name_{i}")
        st.markdown(f"#### {name}")
        col1, col2 = st.columns(2)

        with col1:
            uploaded = st.file_uploader(f"Upload CSV for {name}", type="csv", key=f"file_{i}")

        with col2:
            model = st.selectbox(f"Model for {name}", model_options, key=f"model_{i}")

        category_info.append((name, uploaded, model)) #name of the cateogory, 
        st.markdown("---")

    # -------------------- STACKING --------------------
    st.markdown("### 🚀 Step 3: Stack Models & Save")

    if "stack_ready" not in st.session_state:
        st.session_state.stack_ready = False

    if st.button("Stack!"):
        if any(file is None for (_, file, _) in category_info):
            st.warning("⚠️ Please upload a CSV for every category before stacking.")
            st.session_state.stack_ready = False  # important!
        else:
            st.session_state.stack_ready = True


    if st.session_state.stack_ready:
        model_name = st.text_input("Enter a name for your stacked model:")
        final_model_choice = st.selectbox("Choose final stacking model (meta-estimator):", model_options, key="final_estimator")

        # Step 3: Training logic
    if st.button("Confirm and Train Model"):
    # Input validation
        if not model_name.strip():
            st.error("❗ Please enter a valid name for your stacked model.")
        elif any(file is None for file in [file for (_, file, _) in category_info]):
            st.error("❗ Please upload a CSV file for every category.")
        else:
            with st.spinner("Training and stacking models..."):
                uploaded_files = [file for (_, file, _) in category_info]
                model_choices = [model for (_, _, model) in category_info]
                category_names = [name for (name, _, _) in category_info]

                success = train_and_save_stacked_model(
                    uploaded_files,
                    model_choices,
                    model_name.strip(),   # Cleaned model name
                    final_model_choice, 
                    category_names 
                )

                if success:
                    st.success(f"✅ Model `{model_name}` saved successfully in `/models`.")
                    st.session_state.stack_ready = False
                else:
                    st.error("❌ Something went wrong during training. Please check your files and try again.")

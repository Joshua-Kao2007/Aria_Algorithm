import streamlit as st
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Constants
MODEL_DIR = "models"

# Streamlit page setup
st.set_page_config(page_title="View Trained Models", layout="wide")
st.title("View Trained Models")

# Load available model files
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl") and "_X_test" not in f and "_y_test" not in f]

if not model_files:
    st.warning("⚠️ No models found in the /models directory.")
    st.stop()

# Dropdown for model selection
selected_model_file = st.selectbox("📂 Select a model to view:", sorted(model_files))

if st.button("🔍 Submit to View Model Performance"):
    model_name = selected_model_file.replace(".pkl", "")
    model_path = os.path.join(MODEL_DIR, selected_model_file)

    print(f"Loading model from: {model_path}")
    
    # Try loading unified model dictionary format
    try:
        with open(model_path, "rb") as f:
            model_data = joblib.load(f)


        if isinstance(model_data, dict) and "model" in model_data:
            model = model_data["model"]
            X_test = model_data.get("X_test")
            y_test = model_data.get("y_test")
        else:
            st.error("❌ Model file is not in the expected dictionary format.")
            st.stop()

        if X_test is None or y_test is None:
            st.error("❌ Model file is missing X_test or y_test.")
            st.stop()

        # Run predictions
        y_pred = model.predict(X_test)

        # Display header info
        st.header(f"📌 Model: `{model_name}`")
        st.markdown(f"**🔢 Test Set Size:** {len(y_test)} rows")
        st.markdown(f"**🧮 Number of Features:** {X_test.shape[1]}")
        st.markdown("---")

        # Classification Report
        st.subheader("📊 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

        # Confusion Matrix
        st.subheader("🧾 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
        st.pyplot(fig)

        # Preview of test data
        st.subheader("🔍 Test Data with Predictions")
        preview_df = X_test.copy()
        preview_df["Actual"] = y_test.values
        preview_df["Predicted"] = y_pred
        st.dataframe(preview_df.head(20), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Failed to load or process model: {e}")

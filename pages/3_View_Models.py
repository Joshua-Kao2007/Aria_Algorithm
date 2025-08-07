import streamlit as st
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
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

REVERSE_LABEL_MAP = {0: "Under-Patron", 1: "Patron+"}
  

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
            X_train = model_data.get("X_train")
            y_train = model_data.get("y_train")
            X_test = model_data.get("X_test")
            y_test = model_data.get("y_test")
        else:
            st.error("❌ Model file is not in the expected dictionary format.")
            st.stop()
        
        if X_train is None or y_train is None:
            st.error("❌ Model file is missing X_train or y_train.")
            st.stop()

        if X_test is None or y_test is None:
            st.error("❌ Model file is missing X_test or y_test.")
            st.stop()

        # Run predictions
        y_pred = model.predict(X_test)

        # Display header info
        st.header(f"📌 Model: `{model_name}`")
        st.markdown(f"Training Set Size: {len(y_train)} rows")

        st.markdown(f"**Test Set Size:** {len(y_test)} rows")
        st.markdown(f"**Number of Features:** {X_test.shape[1]}")
        st.markdown("---")

        with st.expander("Show full training set"):
            st.subheader("Trained Parameters")
            st.dataframe(X_train, use_container_width=True)
            st.subheader("Trained Outputs")
            st.dataframe(pd.DataFrame(y_train.map(REVERSE_LABEL_MAP), columns=["Donor_Category"]), use_container_width=True)

 
        # Classification Report
        y_test_named = y_test.map(REVERSE_LABEL_MAP)
        y_pred_named = pd.Series(y_pred, index=y_test.index).map(REVERSE_LABEL_MAP)

        precision = precision_score(y_test_named, y_pred_named, pos_label="Patron+")
        recall = recall_score(y_test_named, y_pred_named, pos_label="Patron+")
        f1 = f1_score(y_test_named, y_pred_named, pos_label="Patron+")
        accuracy = accuracy_score(y_test_named, y_pred_named)

        # Display
        st.subheader("📊 Model Performance Metrics")
        st.markdown(f"**Precision:** {precision:.3f}")
        st.markdown(f"**Recall:** {recall:.3f}")
        st.markdown(f"**F1 Score:** {f1:.3f}")
        st.markdown(f"**Accuracy:** {accuracy:.3f}")

        # Confusion Matrix
        st.subheader("🧾 Confusion Matrix")
        cm = confusion_matrix(y_test_named, y_pred_named, labels = ["Under-Patron", "Patron+"])
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Under-Patron", "Patron+"])
        disp.plot(ax=ax, cmap="Blues")
        st.pyplot(fig)

        # Preview of test data
        st.subheader("🔍 Test Data with Predictions")
        preview_df = X_test.copy()
        preview_df["Actual"] = y_test_named
        preview_df["Predicted"] = y_pred_named
        st.dataframe(preview_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Failed to load or process model: {e}")

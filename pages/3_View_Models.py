import streamlit as st
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

st.set_page_config(page_title="View Models", layout="wide")
st.title("📂 View Saved Models")

# Step 1: Get all .pkl models in the models folder
model_files = [f for f in os.listdir("models") if f.endswith(".pkl")]

if not model_files:
    st.warning("No models found in `/models` folder.")
    st.stop()

# Step 2: Dropdown to select a model
selected_model = st.selectbox("Select a saved model:", model_files)

# Step 3: Upload a test CSV
uploaded_data = st.file_uploader("Upload a CSV for evaluation (must contain 'donor_category')", type="csv")

if st.button("Generate Model Statistics"):
    if uploaded_data is None:
        st.error("Please upload a CSV file to test the model.")
        st.stop()

    # Load model
    model_path = os.path.join("models", selected_model)
    model = joblib.load(model_path)

    # Load and process data
    df = pd.read_csv(uploaded_data)

    if "Donor_Category" not in df.columns:
        st.error("CSV must contain 'Donor_Category' column.")
        st.stop()

    X = df.drop(columns=["Donor_Category"])
    y_true = df["Donor_Category"]

    try:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

        # Display feature columns
        st.subheader("🧠 Feature Columns")
        st.write(X.columns.tolist())

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
        rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)

        st.subheader("📈 Evaluation Metrics")
        st.markdown(f"- **Accuracy**: `{acc:.2f}`")
        st.markdown(f"- **Precision**: `{prec:.2f}`")
        st.markdown(f"- **Recall**: `{rec:.2f}`")
        st.markdown(f"- **F1 Score**: `{f1:.2f}`")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        st.subheader("🔍 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ROC + AUC
        if y_prob is not None:
            st.subheader("📉 ROC Curve + AUC")
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = roc_auc_score(y_true, y_prob)

            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
            ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.legend()
            st.pyplot(fig2)

        else:
            st.info("This model does not support probability outputs for ROC/AUC.")

    except Exception as e:
        st.error(f"❌ Error generating statistics: {e}")

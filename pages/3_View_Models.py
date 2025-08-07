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
st.title("📂 View Saved Stacked Models")

# Step 1: Get all saved .pkl models
model_files = [f for f in os.listdir("models") if f.endswith(".pkl")]

if not model_files:
    st.warning("No models found in `/models` folder.")
    st.stop()

# Step 2: Select a model
selected_model = st.selectbox("Select a saved model:", model_files)

# Step 3: Upload test data
uploaded_data = st.file_uploader("Upload a test CSV (must include 'Donor_Category' as label)", type="csv")

if st.button("Generate Model Statistics"):
    if uploaded_data is None:
        st.error("Please upload a CSV file.")
        st.stop()

    # Step 4: Load model
    model_path = os.path.join("models", selected_model)
    model = joblib.load(model_path)

    # Step 5: Load and validate CSV
    df = pd.read_csv(uploaded_data)

    if "Donor_Category" not in df.columns:
        st.error("CSV must contain a 'Donor_Category' column as the target label.")
        st.stop()

    if "customer_no" in df.columns:
        ids = df["customer_no"]
        df = df.drop(columns = ['customer_no'])

    X = df.drop(columns=["Donor_Category"])
    y_true = df["Donor_Category"]

    # Step 6: Check for missing columns in base models
    missing = []
    for name, estimator in model.named_estimators_.items():
        if hasattr(estimator, "feature_names_in_"):
            required_cols = list(estimator.feature_names_in_)
            missing_cols = [col for col in required_cols if col not in X.columns]
            if missing_cols:
                missing.append((name, missing_cols))

    if missing:
        st.error("❌ Some base models are missing required input features:")
        for model_name, cols in missing:
            st.markdown(f"- **{model_name}** is missing: `{', '.join(cols)}`")
        st.stop()

    # Step 7: Run predictions
    try:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

        # Step 8: Show feature list
        st.subheader("🧠 Features Used")
        st.write(X.columns.tolist())

        # Step 9: Evaluation metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label="Patron+", zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label="Patron+", zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label="Patron+", zero_division=0)

        st.subheader("📊 Evaluation Metrics")
        st.markdown(f"- **Accuracy**: `{acc:.2f}`")
        st.markdown(f"- **Precision**: `{prec:.2f}`")
        st.markdown(f"- **Recall**: `{rec:.2f}`")
        st.markdown(f"- **F1 Score**: `{f1:.2f}`")

        # Step 10: Confusion matrix with custom labels
        cm = confusion_matrix(y_true, y_pred, labels=["Under-Patron", "Patron+"])
        st.subheader("🔍 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Under-Patron", "Patron+"],
                    yticklabels=["Under-Patron", "Patron+"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Step 11: ROC Curve + AUC
        if y_prob is not None:
            st.subheader("📉 ROC Curve + AUC")
            fpr, tpr, _ = roc_curve(y_true == "Patron+", y_prob)
            auc_score = roc_auc_score(y_true == "Patron+", y_prob)

            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
            ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.legend()
            st.pyplot(fig2)
        else:
            st.info("ℹ️ This model does not support probability outputs (e.g. no ROC/AUC).")

    except Exception as e:
        st.error(f"❌ Error during evaluation: {e}")

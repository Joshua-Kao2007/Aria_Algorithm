import pandas as pd
import os
import joblib
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Map dropdown model names to sklearn classes
MODEL_MAP = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "AdaBoost": AdaBoostClassifier,
    "KNN": KNeighborsClassifier,
}

def train_and_save_stacked_model(uploaded_files, model_choices, model_name, final_estimator_str):
    X_all = []
    y_all = []
    estimators = []

    try:
        for i, (file, model_name_str) in enumerate(zip(uploaded_files, model_choices)):
            df = pd.read_csv(file)

            if "Donor_Category" not in df.columns:
                raise ValueError(f"'donor_category' column not found in uploaded file {i+1}")

            X = df.drop(columns=["Donor_Category"])
            y = df["Donor_Category"]

            model_cls = MODEL_MAP[model_name_str]
            model = model_cls()
            model.fit(X, y)

            X_all.append(X)
            y_all.append(y)
            estimators.append((f"model_{i}", model))

        # Combine all data for stacking
        X_stack = pd.concat(X_all, axis=0)
        y_stack = pd.concat(y_all, axis=0)

        # Use selected model as final estimator
        final_cls = MODEL_MAP[final_estimator_str]
        final_estimator = final_cls()

        final_model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            passthrough=False
        )

        final_model.fit(X_stack, y_stack)

        # Save the stacked model
        os.makedirs("models", exist_ok=True)
        joblib.dump(final_model, f"models/{model_name}.pkl")
        return True

    except Exception as e:
        print(f"Error training stacked model: {e}")
        return False

import pandas as pd
import os
import joblib
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from utils.evaluation_utils import evaluate_model
from utils.model_imputation import clean_and_impute
from utils.merge_utils import merge_all_files

MODEL_MAP = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "AdaBoost": AdaBoostClassifier,
    "KNN": KNeighborsClassifier,
    "LightGBM": LGBMClassifier,
    "XGBoost": XGBClassifier,
    "Linear Regression": LinearRegression,
}

LABEL_MAP = {"Under-Patron": 0, "Patron+": 1}

def train_and_save_stacked_model(uploaded_files, model_choices, model_name, final_estimator_str, category_names):
    try:
        print("HIII")
        # Merge all uploaded files by customer_no
        df = merge_all_files(uploaded_files)
        # Clean and impute
        df = clean_and_impute(df)

        # Prepare training data
        X = df.drop(columns=["Donor_Category", "customer_no"], errors="ignore")
        y = df["Donor_Category"].map(LABEL_MAP)

        estimators = []
        for i, model_name_str in enumerate(model_choices):
            model_cls = MODEL_MAP[model_name_str]
            model = model_cls()
            model.fit(X, y)
            estimators.append((f"model_{i}", model))

        final_cls = MODEL_MAP[final_estimator_str]
        final_estimator = final_cls()
        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            passthrough=False,
            cv=3
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        stacking_model.fit(X_train, y_train)


        print("GOT HERE")
        # Save everything
        os.makedirs("models", exist_ok=True)
        joblib.dump({
            "model": stacking_model,
            "X_test": X_test,
            "y_test": y_test,
            "columns": X.columns.tolist(),
            "category_models": model_choices,
            "category_names": [f"model_{i}" for i in range(len(model_choices))]
        }, f"models/{model_name}.pkl")
        print("Model saved successfully.")

        print(f"✅ Stacked model '{model_name}' trained and saved successfully.")
        return True

    except Exception as e:
        print(f"❌ Error training stacked model: {e}")
        return False

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
    """
    Train a stacked model from multiple category datasets and save it to disk.

    Inner models are set to medium strength (e.g., tree depth scaled to dataset size).
    Final meta-model is heavily boosted with many iterations for maximum performance.
    """
    try:
        # Merge all uploaded files by customer_no
        df = merge_all_files(uploaded_files)
        
        # Clean and impute missing values
        df = clean_and_impute(df)

        # Prepare features and labels
        X = df.drop(columns=["Donor_Category", "customer_no"], errors="ignore")
        y = df["Donor_Category"].map(LABEL_MAP)

        # Determine dataset size for dynamic tuning
        n_rows = X.shape[0]

        # --- Build base estimators with "medium" complexity ---
        estimators = []
        for i, model_name_str in enumerate(model_choices):
            model_cls = MODEL_MAP[model_name_str]

            # Default parameters
            params = {}

            if model_name_str == "Random Forest":
                params = {
                    "n_estimators": 200,
                    "max_depth": min(max(3, n_rows // 3), 15),
                    "random_state": 42
                }
            elif model_name_str == "Logistic Regression":
                params = {
                    "max_iter": 500
                }
            elif model_name_str == "KNN":
                params = {
                    "n_neighbors": min(15, max(3, n_rows // 10))
                }
            elif model_name_str == "XGBoost":
                params = {
                    "max_depth": min(max(3, n_rows // 3), 15),
                    "n_estimators": 200,
                    "learning_rate": 0.1,
                    "random_state": 42,
                    "use_label_encoder": False,
                    "eval_metric": "logloss"
                }
            elif model_name_str == "LightGBM":
                params = {
                    "max_depth": min(max(3, n_rows // 3), 15),
                    "n_estimators": 200,
                    "learning_rate": 0.1,
                    "random_state": 42
                }
            elif model_name_str == "AdaBoost":
                params = {
                    "n_estimators": 200,
                    "learning_rate": 1.0,
                    "random_state": 42
                }

            model = model_cls(**params)
            model.fit(X, y)
            estimators.append((f"model_{i}", model))

        # --- Build final meta-estimator with "tons of iterations" ---
        final_cls = MODEL_MAP[final_estimator_str]
        final_params = {}

        if final_estimator_str == "AdaBoost":
            final_params = {
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "random_state": 42
            }
        elif final_estimator_str == "XGBoost":
            final_params = {
                "n_estimators": 2000,
                "learning_rate": 0.05,
                "max_depth": min(max(3, n_rows // 3), 15),
                "random_state": 42,
                "use_label_encoder": False,
                "eval_metric": "logloss"
            }
        elif final_estimator_str == "LightGBM":
            final_params = {
                "n_estimators": 2000,
                "learning_rate": 0.05,
                "max_depth": min(max(3, n_rows // 3), 15),
                "random_state": 42
            }

        final_estimator = final_cls(**final_params)

        # --- Stacking classifier ---
        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            passthrough=False,
            cv=3
        )

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train stacking model
        stacking_model.fit(X_train, y_train)

        # Save everything
        os.makedirs("models", exist_ok=True)
        joblib.dump({
            "model": stacking_model,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "columns": X.columns.tolist(),
            "category_models": model_choices,
            "category_names": [f"model_{i}" for i in range(len(model_choices))]
        }, f"models/{model_name}.pkl")

        print(f"✅ Stacked model '{model_name}' trained and saved successfully.")
        return True

    except Exception as e:
        print(f"❌ Error training stacked model: {e}")
        return False

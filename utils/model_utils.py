import pandas as pd
import os
import joblib
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from utils.evaluation_utils import evaluate_model
from utils.model_imputation import clean_and_impute

MODEL_MAP = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "AdaBoost": AdaBoostClassifier,
    "KNN": KNeighborsClassifier,
    "LightGBM": LGBMClassifier,
    "XGBoost": XGBClassifier,
}

LABEL_MAP = {"Under-Patron": 0, "Patron+": 1}

def train_and_save_stacked_model(uploaded_files, model_choices, model_name, final_estimator_str):
    X_all = []
    y_all = []
    estimators = []

    try:
        for i, (file, model_name_str) in enumerate(zip(uploaded_files, model_choices)):
            print("HI")
            df = pd.read_csv(file)
            if "customer_no" in df.columns:
                df.drop(columns=["customer_no"], inplace=True)
            if "Donor_Category" not in df.columns:
                raise ValueError(f"'Donor_Category' column not found in uploaded file {i+1}")
            
            print(df.head())
            print(df.shape)

            df = clean_and_impute(df)


            print(f"✅ File {i+1} cleaned and imputed successfully.")
            X = df.drop(columns=["Donor_Category"])
            y = df["Donor_Category"].map(LABEL_MAP)
            model_cls = MODEL_MAP[model_name_str]
            model = model_cls()
            model.fit(X, y)
            estimators.append((f"model_{i}", model))
            X_all.append(X)
            y_all.append(y) 


        X_stack = pd.concat(X_all, axis=0)
        y_stack = pd.concat(y_all, axis=0)

        final_cls = MODEL_MAP[final_estimator_str]
        final_estimator = final_cls()
        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            passthrough=False,
            cv=3
        )
        stacking_model.fit(X_stack, y_stack)
        os.makedirs("models", exist_ok=True)
        joblib.dump(stacking_model, f"models/{model_name}.pkl")

        print(f"✅ Stacked model '{model_name}' trained and saved successfully.")  
        
    except Exception as e:
        print("HELLO")
        print(f"❌ Error training stacked model: {e}")
        return False
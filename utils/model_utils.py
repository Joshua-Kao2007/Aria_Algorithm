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
    X_all = []
    y_all = []
    estimators = []

    try:
        for i, (file, model_name_str) in enumerate(zip(uploaded_files, model_choices)):
            print("HI")
            df = pd.read_csv(file)
            if "Donor_Category" not in df.columns:
                raise ValueError(f"'Donor_Category' column not found in uploaded file {i+1}")
            print(df.head())
            print(df.shape)
            




            df = clean_and_impute(df)


            print(f"✅ File {i+1} cleaned and imputed successfully." + df.head().to_string())



            X = df.drop(columns=["Donor_Category"])
            y = df["Donor_Category"].map(LABEL_MAP)
            model_cls = MODEL_MAP[model_name_str]
            model = model_cls()
            model.fit(X, y)
            estimators.append((f"model_{i}", model))
            X_all.append(X)
            y_all.append(y) 


        X_stack = pd.concat(X_all, axis=1)
        y_stack = pd.concat(y_all, axis=1)

        final_cls = MODEL_MAP[final_estimator_str]
        final_estimator = final_cls()
        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            passthrough=False,
        )
        print(stacking_model.get_params())
        print(X_stack.shape, y_stack.shape)
        print(X_stack.head())
        print(y_stack.head())
        stacking_model.fit(X_stack, y_stack)
        print(f"✅ Stacked model '{model_name}' trained successfully.")

        X_train, X_test, y_train, y_test = train_test_split(X_stack, y_stack, test_size=0.25, random_state=42)
        category_names = [os.path.splitext(os.path.basename(f.name))[0] for f in uploaded_files]


        os.makedirs("models", exist_ok=True)
        joblib.dump({
            "model": stacking_model,
            "X_test": X_test,
            "y_test": y_test,
            "columns": X_stack.columns.tolist(),
            "category_models": model_choices,
            "category_names": category_names,
            "X_sample": X_stack.head(100),
        }, f"models/{model_name}.pkl")

        print(f"✅ Stacked model '{model_name}' trained and saved successfully.")  
        
    except Exception as e:
        print("HELLO")
        print(f"❌ Error training stacked model: {e}")
        return False
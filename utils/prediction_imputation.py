import pandas as pd
from sklearn.impute import SimpleImputer

def impute_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop label or ID columns if present
    for col in ["Donor_Category", "customer_no"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(exclude=["number"]).columns

    # Impute numeric columns with mean
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy="mean")
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    # Impute categorical columns with mode
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    return df

from sklearn.impute import SimpleImputer
import pandas as pd

LABEL_MAP = {"Under-Patron": 0, "Patron+": 1}

def clean_and_impute(df):
    # Only keep valid Donor_Category rows
    print("Initial shape in clean_and_impute:", df.shape)

    df = df[df["Donor_Category"].isin(LABEL_MAP.keys())]
    print("Shape after filtering Donor_Category:", df.shape)

    # Strip whitespace from all string cells
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    print("Shape after stripping whitespace:", df.shape)


    # Separate into donor groups
    patron_df = df[df["Donor_Category"] == "Patron+"].copy()
    nonpatron_df = df[df["Donor_Category"] == "Under-Patron"].copy()
    print("Shape of Patron+ group:", patron_df.shape)

    # Function to impute one group's data
    def impute_group(group_df):
        numeric_cols = group_df.select_dtypes(include=["number"]).columns
        categorical_cols = group_df.select_dtypes(include=["object", "category"]).columns.drop("Donor_Category")

        # Impute numeric columns
        if not numeric_cols.empty:
            num_imputer = SimpleImputer(strategy="median")
            group_df[numeric_cols] = num_imputer.fit_transform(group_df[numeric_cols])

        # Impute categorical columns
        if not categorical_cols.empty:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            group_df[categorical_cols] = cat_imputer.fit_transform(group_df[categorical_cols])

        return group_df

    # Impute both groups
    print("Shape before imputation - Patron+:", patron_df.shape)
    patron_df = impute_group(patron_df)
    nonpatron_df = impute_group(nonpatron_df)
    print("Shape after imputation - Patron+:", patron_df.shape)

    # Recombine them
    df_cleaned = pd.concat([patron_df, nonpatron_df], axis=0).reset_index(drop=True)
    print(df_cleaned.tail(20))

    return df_cleaned

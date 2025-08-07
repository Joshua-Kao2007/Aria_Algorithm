# merge_utils.py
import pandas as pd

def merge_all_files(uploaded_files):
    merged = None
    for file in uploaded_files:
        df = pd.read_csv(file)
        df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

        if merged is None:
            merged = df
            print(merged["customer_no"].head())
        else:
            print("BYYYE")
            print(df["customer_no"].head())
            print(merged["customer_no"].head())

            merged = pd.merge(merged, df, on="customer_no", how="outer", suffixes=('', '_dup'))
            print("HELLO", merged.shape)
            print("HEAD", merged.head())
            
            # Remove duplicate donor_category columns, keeping the first non-null
            donor_cols = [col for col in merged.columns if "Donor_Category" in col]
            if len(donor_cols) > 1:
                merged["Donor_Category"] = merged[donor_cols].bfill(axis=1).iloc[:, 0]
                merged.drop(columns=[col for col in donor_cols if col != "Donor_Category"], inplace=True)

    return merged

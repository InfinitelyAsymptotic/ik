import pandas as pd, os

def save_reference_data(df, path="monitoring/reference.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

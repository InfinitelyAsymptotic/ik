"""Minimal preprocessing using config vars"""
import os
import pandas as pd

input_path = os.getenv("RAW_FILE", "data/ieee_fraud/train_transaction.csv")
output_path = os.getenv("CLEANED_FILE", "data/cleaned/train_transaction_cleaned.csv")

os.makedirs(os.path.dirname(output_path), exist_ok=True)
raw = pd.read_csv(input_path, nrows=2)
raw.dropna().reset_index(drop=True).to_csv(output_path, index=False)

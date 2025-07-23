# download_data.py
import os
import pandas as pd
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_kaggle_dataset():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    dataset_path = "data/raw"
    print("📥 Downloading dataset...")
    api.competition_download_files("ieee-fraud-detection", path=dataset_path)

    with zipfile.ZipFile(os.path.join(dataset_path, "ieee-fraud-detection.zip"), "r") as z:
        z.extractall(dataset_path)
    print("✅ Dataset extracted")

def preprocess_data():
    print("🔄 Preprocessing...")
    trans_path = "data/raw/train_transaction.csv"
    ident_path = "data/raw/train_identity.csv"

    df_trans = pd.read_csv(trans_path)
    df_ident = pd.read_csv(ident_path)

    df = df_trans.merge(df_ident, on="TransactionID", how="left")
    df = df.fillna(0)

    # Select numeric features + target for fast training
    features = ["TransactionAmt", "dist1", "card1", "card2", "isFraud"]
    df = df[features].dropna()

    df.to_csv("data/kaggle_fraud_processed.csv", index=False)
    print(f"✅ Preprocessed data saved: data/kaggle_fraud_processed.csv")

if __name__ == "__main__":
    download_kaggle_dataset()
    preprocess_data()

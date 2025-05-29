# 🧠 AWS-Based ML Data Management Demo

This project demonstrates a modular, automated pipeline for managing ML datasets using AWS services like EC2, S3, DVC, and Feast. It handles ingestion, preprocessing, versioning, and feature storage using best practices.

---

## 🎯 Objective

Build an end-to-end data management system for the [IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/competitions/ieee-fraud-detection) with:

- S3 for storage (raw/cleaned)
- DVC for data versioning
- Feast for feature management
- Central configuration for easy reuse

---

## ✅ Prerequisites

- AWS account with billing enabled
- IAM user with required permissions
- EC2 (Ubuntu 22.04) instance
- SSH or Session Manager access
- [Kaggle API key (`kaggle.json`)](https://www.kaggle.com/docs/api)
- Git, DVC, Git LFS, Python 3.8+

---

## 🏗️ Project Structure

```
ik-dataManagement/
├── bootstrap.sh           # Main automation script
├── config.sh              # Central configuration (edit this)
├── scripts/
│   ├── preprocess.py      # Cleans data
│   └── feature_repo.py    # Feast feature definitions
└── README.md
```

---

## 🚀 Quickstart

### 1. 🖥️ Launch EC2

- Ubuntu 22.04
- Attach IAM role with:
  - AmazonS3FullAccess
  - AmazonSageMakerFullAccess
  - CloudWatchFullAccess
  - AmazonSSMManagedInstanceCore
- Add tag: `Purpose=data-management-demo`

### 2. 📂 Upload Files

Upload or clone this repo onto your EC2 instance:

```bash
git clone -b dataManagement https://github.com/InfinitelyAsymptotic/ik.git
cd ik
```

Ensure your `~/.kaggle/kaggle.json` file exists on the EC2.

### 3. ⚙️ Configure

Edit `config.sh` to set your user name:

```bash
nano config.sh
```

Update:
```bash
export USER_TAG="yourname"  # change this
```

### 4. 🧰 Run Setup

```bash
chmod +x bootstrap.sh
source bootstrap.sh
```

This will:
- Install dependencies
- Download and unzip the Kaggle dataset
- Upload to S3
- Initialize Git + DVC, push data
- Preprocess and version cleaned data
- Set up and materialize Feast feature store

---

## 📊 Using Feast

Query features in Python:

```python
from feast import FeatureStore

store = FeatureStore(repo_path="fraud_demo")
df = store.get_historical_features(
    entity_df="SELECT TransactionID, TransactionDT FROM transactions LIMIT 5",
    features=["transaction_features.TransactionAmt"]
).to_df()

print(df)
```

---

## 🧠 Architecture

```text
Kaggle → EC2 → S3 (Raw/Cleaned)
                     ↓
                  DVC (versioned)
                     ↓
              Feast (feature store)
```

---

## 📌 Notes

- All custom variables (bucket, paths, user tag) are centralized in `config.sh`
- Re-run `bootstrap.sh` after changing configs or adding features
- Add more preprocessing or ETL logic inside `scripts/`

---


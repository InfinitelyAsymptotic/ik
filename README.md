# 🚀 Fraud Detection Model Deployment with FastAPI

This project provides a complete workflow to **train and deploy a machine learning fraud detection model** using the IEEE-CIS Fraud Detection Kaggle dataset, with a FastAPI-based REST API for real-time inference.

## 📦 Project Structure

```
fraud-api/
├── api.py
├── download_data.py
├── train_model.py
├── data/
│   ├── kaggle_fraud_processed.csv
│   └── raw/
├── models/
│   └── fraud_model.joblib
├── requirements.txt
└── README.md
```

## 📝 Features

* **Kaggle Data Integration:** Automated download and preprocessing of the IEEE-CIS Fraud Detection dataset.
* **Model Training:** Easily train a Random Forest model with reproducible results.
* **Model Serialization:** Persist your trained model for later use.
* **FastAPI Inference API:** Serve predictions over HTTP locally.
* **Reproducible Pipeline:** Clear separation of data handling, model training, and deployment.
* **Minimal Dependencies:** Installs quickly on any machine.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd fraud-api
```

### 2. Install Dependencies

Create a virtual environment (recommended), then install:

```bash
pip install -r requirements.txt
```

### 3. Configure Kaggle Credentials

Download your `kaggle.json` from your Kaggle Account Settings and place it at `~/.kaggle/kaggle.json`. Set permissions if necessary:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Download & Preprocess Kaggle Data

```bash
python download_data.py
```

After running, your directory will include `data/kaggle_fraud_processed.csv`.

### 5. Train the Model

```bash
python train_model.py
```

Model metrics and a serialized model `fraud_model.joblib` will be saved in the `models/` folder.

### 6. Run FastAPI Inference Server

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## ⚡ Prediction API Guide

### Endpoint

* **POST** `/predict`

### Request Payload (JSON)

```json
{
  "TransactionAmt": 100.0,
  "dist1": 25.0,
  "card1": 1580,
  "card2": 365.0
}
```

### Response

```json
{
  "isFraud": 0
}
```

**0:** Not fraud  
**1:** Fraud

### Example cURL Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"TransactionAmt": 100.0, "dist1": 50.0, "card1": 1500, "card2": 300.0}'
```

## 🗂️ Scripts Overview

| Script | Role |
|--------|------|
| `download_data.py` | Download & preprocess Kaggle dataset |
| `train_model.py` | Train and save the Random Forest model |
| `api.py` | Start FastAPI server for predictions |
| `requirements.txt` | Python dependencies |
# # api.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# import numpy as np
# import joblib
# import os

# app = FastAPI()

# MODEL_PATH = "models/fraud_model.joblib"
# model = None

# class FraudRequest(BaseModel):
#     TransactionAmt: float
#     dist1: float
#     card1: float
#     card2: float

# @app.on_event("startup")
# def startup_event():
#     global model
#     if os.path.exists(MODEL_PATH):
#         model = joblib.load(MODEL_PATH)
#         print("✅ Model loaded")
#     else:
#         raise RuntimeError("Model file not found. Train it first.")

# @app.post("/predict")
# def predict_fraud(request: FraudRequest):
#     features = np.array([[request.TransactionAmt, request.dist1, request.card1, request.card2]])
#     prediction = model.predict(features)
#     return {"isFraud": int(prediction[0])}

# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI()

MODEL_PATH = "models/fraud_model.joblib"
model = None

class FraudRequest(BaseModel):
    TransactionAmt: float
    dist1: float
    card1: float
    card2: float

# ADDED: Safe model loading function
def load_model_safely():
    """Load model with NumPy compatibility error handling."""
    try:
        return joblib.load(MODEL_PATH)
    except ModuleNotFoundError as e:
        if 'numpy._core' in str(e):
            raise RuntimeError(
                "NumPy version mismatch. Please:\n"
                "1. pip install --upgrade numpy joblib scikit-learn\n"
                "2. Re-train model: python train_model.py"
            )
        raise

@app.on_event("startup")
def startup_event():
    global model
    if os.path.exists(MODEL_PATH):
        model = load_model_safely()  # CHANGED: Use safe loader
        print("✅ Model loaded")
    else:
        raise RuntimeError("Model file not found. Train it first.")

@app.post("/predict")
def predict_fraud(request: FraudRequest):
    features = np.array([[request.TransactionAmt, request.dist1, request.card1, request.card2]])
    prediction = model.predict(features)
    return {"isFraud": int(prediction[0])}
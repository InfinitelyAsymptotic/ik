from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib, glob, os

from inference_utils import FraudModelPreprocessor  # Ensure this exists and matches training

app = FastAPI()

class FraudInput(BaseModel):
    TransactionAmt: float
    dist1: float
    card1: float
    card2: float

MODEL_PATH = "models"
PREPROCESSOR_PATH = "models"
FEATURE_NAMES_FILE = None  # Will auto-load latest if not set explicitly

model = None
preprocessor = None
feature_names = []

@app.on_event("startup")
def load_model_and_preprocessor():
    global model, preprocessor, feature_names

    # Load most recent model
    models = sorted(glob.glob(os.path.join(MODEL_PATH, "fraud_model_*.joblib")), reverse=True)
    if not models:
        raise RuntimeError("❌ No model found. Run training first.")
    model = joblib.load(models[0])
    print(f"✅ Model loaded: {models[0]}")

    # Load corresponding preprocessor
    preprocessors = sorted(glob.glob(os.path.join(PREPROCESSOR_PATH, "fraud_preprocessor_*.joblib")), reverse=True)
    if not preprocessors:
        raise RuntimeError("❌ No preprocessor found. Ensure training saves it.")
    preprocessor = joblib.load(preprocessors[0])
    print(f"✅ Preprocessor loaded: {preprocessors[0]}")

    # Load corresponding feature names list
    feature_files = sorted(glob.glob(os.path.join(PREPROCESSOR_PATH, "feature_names_*.txt")), reverse=True)
    if not feature_files:
        raise RuntimeError("❌ Feature names file not found. Ensure training saves it.")
    with open(feature_files[0], 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    print(f"✅ Total expected features: {len(feature_names)}")

@app.post("/predict")
def predict(inp: FraudInput):
    try:
        input_data = inp.dict()

        # Reconstruct feature vector using full feature names, fill missing with 0
        full_input = {col: input_data.get(col, 0) for col in feature_names}
        df = pd.DataFrame([full_input])

        # Apply preprocessor (if needed, else you can skip if identity)
        vector = preprocessor.transform(df)

        pred = model.predict(vector)[0]
        return {"isFraud": int(pred)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

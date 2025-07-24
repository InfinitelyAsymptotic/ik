from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, os, glob, pandas as pd
from inference_utils import FraudModelPreprocessor
from monitoring_data_logger import append_input
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()

class FraudInput(BaseModel):
    TransactionAmt: float
    dist1: float
    card1: float
    card2: float

@app.on_event("startup")
def load_artifacts():
    global model, preprocessor, feature_names
    mfile = sorted(glob.glob("models/fraud_model_*.joblib"))[-1]
    pfile = sorted(glob.glob("models/fraud_preprocessor_*.joblib"))[-1]
    model = joblib.load(mfile)
    preprocessor = joblib.load(pfile)
    with open(f"models/feature_names_{mfile.split('_')[-1].replace('.joblib','')}.txt") as f:
        feature_names = [line.strip() for line in f.readlines()]

@app.post("/predict")
def predict(inp: FraudInput):
    input_dict = inp.dict()
    append_input(input_dict)
    X = pd.DataFrame([{k: input_dict.get(k, 0) for k in feature_names}])
    X = preprocessor.transform(X)
    pred = model.predict(X)[0]
    return {"isFraud": int(pred)}


@app.get("/monitoring")
def dashboard():
    json_path = "drift_report.json"
    if not os.path.exists(json_path):
        raise HTTPException(404, detail="Drift report not found")
    with open(json_path) as f:
        data = json.load(f)
    return JSONResponse(content=data)


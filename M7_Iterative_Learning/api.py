from fastapi import FastAPI
from ab_router import route_ab
import joblib, pandas as pd, os
from inference_utils import FraudModelPreprocessor
from monitoring_data_logger import append_input

app = FastAPI()
model_a = joblib.load('models/fraud_model_A.joblib')
model_b = joblib.load('models/fraud_model_B.joblib')
preprocessor = joblib.load('models/fraud_preprocessor.joblib')
with open('models/feature_names.txt') as f:
    feature_names = [line.strip() for line in f]

def preprocess(inp):  # Ensures the feature order matches training
    df = pd.DataFrame([{k: inp.get(k, 0) for k in feature_names}])
    return preprocessor.transform(df)

def log_for_labeling(record, pred, label_file="label_inbox/to_label.csv"):
    os.makedirs("label_inbox", exist_ok=True)
    rec = dict(record)
    rec["auto_pred"] = int(pred)
    pd.DataFrame([rec]).to_csv(label_file, mode="a", header=not os.path.exists(label_file), index=False)

@app.post("/predict_a")
def predict_a(inp: dict):
    X = preprocess(inp)
    pred = model_a.predict(X)[0]
    append_input(inp)
    log_for_labeling(inp, pred)
    return {"model": "A", "isFraud": int(pred)}

@app.post("/predict_b")
def predict_b(inp: dict):
    X = preprocess(inp)
    pred = model_b.predict(X)[0]
    append_input(inp)
    log_for_labeling(inp, pred)
    return {"model": "B", "isFraud": int(pred)}

@app.post("/predict_ab")
def predict_ab(inp: dict):
    route = route_ab()
    if route == "A":
        return predict_a(inp)
    else:
        return predict_b(inp)

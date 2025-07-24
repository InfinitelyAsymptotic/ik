import pandas as pd, os
import mlflow, joblib
from train import load_and_prepare_data
from inference_utils import FraudModelPreprocessor
from sklearn.ensemble import RandomForestClassifier

def load_new_labels(label_dir="label_inbox/"):
    labeled = [f for f in os.listdir(label_dir) if f.endswith("labeled.csv")]
    if not labeled:
        return None
    dfs = [pd.read_csv(os.path.join(label_dir, f)) for f in labeled]
    return pd.concat(dfs) if dfs else None

def retrain(base_data_path="data/kaggle_fraud.csv"):
    df_train = pd.read_csv(base_data_path)
    new_labels = load_new_labels()
    if new_labels is not None:
        df_train = pd.concat([df_train, new_labels], ignore_index=True)
    X, y = df_train.drop("isFraud", axis=1), df_train["isFraud"]
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X, y)
    joblib.dump(model, "models/fraud_model_A.joblib")
    joblib.dump(model, "models/fraud_model_B.joblib")
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    with mlflow.start_run(run_name="iterative_retrain"):
        mlflow.log_metric("train_size", len(X))
    print("Retrained with new labels. Model saved.")

if __name__ == "__main__":
    retrain()

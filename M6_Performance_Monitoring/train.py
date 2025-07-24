import argparse, os, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import mlflow, joblib
from inference_utils import FraudModelPreprocessor
from monitoring_utils import save_reference_data

def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df = df.dropna(thresh=len(df)*0.5, axis=1).fillna(0)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    if 'isFraud' not in df.columns:
        raise ValueError("isFraud column missing")
    y = df['isFraud']
    X = df.drop('isFraud', axis=1).apply(pd.to_numeric, errors='coerce').fillna(0)

    preprocessor = FraudModelPreprocessor()
    preprocessor.fit(df, 'isFraud')

    return train_test_split(X, y, test_size=0.2, stratify=y), preprocessor, list(X.columns)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--experiment_name', default='fraud_monitoring_model')
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    try: mlflow.create_experiment(args.experiment_name)
    except: pass
    exp = mlflow.get_experiment_by_name(args.experiment_name)

    with mlflow.start_run(experiment_id=exp.experiment_id):
        (X_train, X_test, y_train, y_test), preprocessor, feature_names = load_and_prepare_data(args.data)
        save_reference_data(X_train)

        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }

        mlflow.log_params(vars(args))
        mlflow.log_metrics(metrics)

        run_id = mlflow.active_run().info.run_id
        joblib.dump(model, f"models/fraud_model_{run_id}.joblib")
        preprocessor.save(f"models/fraud_preprocessor_{run_id}.joblib")
        with open(f"models/feature_names_{run_id}.txt", "w") as f:
            f.write("\n".join(feature_names))

if __name__ == "__main__":
    main()

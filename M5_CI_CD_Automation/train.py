import argparse, os, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import mlflow, mlflow.sklearn, joblib

from inference_utils import FraudModelPreprocessor  # Custom class for inference alignment

def load_and_prepare_data(path):
    df = pd.read_csv(path)

    # Drop columns with too many missing values, then fill remaining NAs
    df = df.dropna(thresh=len(df) * 0.5, axis=1).fillna(0)

    # Label encode object columns (except target column)
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object' and column != 'isFraud':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            label_encoders[column] = le

    if 'isFraud' not in df.columns:
        raise ValueError("Target column 'isFraud' not found in the dataset")

    y = df['isFraud']
    X = df.drop('isFraud', axis=1)

    # Convert all features to numeric, coercing non-numeric to NaN -> will be filled
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X = X.fillna(0)

    print(f"✅ Dataset shape: {X.shape}")
    print(f"📌 Features: {list(X.columns)}")
    print(f"🎯 Target distribution: {y.value_counts().to_dict()}")

    # Fit preprocessor on full training DataFrame (X + y)
    full_df = X.copy()
    full_df['isFraud'] = y
    preprocessor = FraudModelPreprocessor()
    preprocessor.fit(full_df, target_col='isFraud')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor, list(X.columns)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--experiment_name', default='fraud_detection')
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"❌ Data file not found: {args.data}")

    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))

    try:
        mlflow.create_experiment(args.experiment_name)
    except:
        pass

    exp = mlflow.get_experiment_by_name(args.experiment_name)

    with mlflow.start_run(experiment_id=exp.experiment_id):
        X_train, X_test, y_train, y_test, preprocessor, feature_columns = load_and_prepare_data(args.data)

        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            class_weight='balanced'
        )

        print("🚀 Training model...")
        clf.fit(X_train, y_train)

        print("🔍 Evaluating model...")
        y_pred = clf.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0)
        }

        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        mlflow.log_params(vars(args))
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(clf, "model")

        # Save artifacts
        run_id = mlflow.active_run().info.run_id
        os.makedirs("models", exist_ok=True)

        model_path = f"models/fraud_model_{run_id}.joblib"
        preprocessor_path = f"models/fraud_preprocessor_{run_id}.joblib"
        feature_names_path = f"models/feature_names_{run_id}.txt"

        joblib.dump(clf, model_path)
        preprocessor.save(preprocessor_path)

        with open(feature_names_path, 'w') as f:
            f.write('\n'.join(feature_columns))

        print(f"💾 Model saved to: {model_path}")
        print(f"🧠 Preprocessor saved to: {preprocessor_path}")
        print(f"📝 Feature names saved to: {feature_names_path}")

if __name__ == "__main__":
    main()

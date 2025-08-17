import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# --- 1. Set up MLflow ---
# This will create a local 'mlruns' directory to store results
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("IEEE Fraud Detection")

# --- 2. Load and Prepare Data ---
print("Loading and preparing data...")
try:
    # Use a smaller sample for a quick training run
    df = pd.read_csv("data/train_transaction.csv", nrows=50000)
    df = df.select_dtypes(include='number').fillna(0) # Simple preprocessing
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except FileNotFoundError:
    print("ERROR: Data not found. Please run 'bash setup.sh' first.")
    exit()

# --- 3. Train and Evaluate Model ---
with mlflow.start_run() as run:
    print(f"Starting MLflow Run: {run.info.run_name}")

    # Log parameters
    params = {"n_estimators": 100, "max_depth": 10}
    mlflow.log_params(params)

    # Train model
    print("Training model...")
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Log metrics and model
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    print("\n✅ Training complete! Run 'mlflow ui' to see the results.")
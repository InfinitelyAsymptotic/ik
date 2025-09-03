import pandas as pd
import ray
import mlflow

from ray.air import session
from ray.air.config import ScalingConfig
from ray.train.xgboost import XGBoostTrainer
from ray.air.integrations.mlflow import MLflowLoggerCallback

# --- 1. Load Data and Convert to Ray Dataset ---
print("Loading data and converting to Ray Dataset...")
try:
    # Use a smaller sample for a quick training run
    df = pd.read_csv("data/train_transaction.csv", nrows=50000)
    df = df.select_dtypes(include='number').fillna(0) # Simple preprocessing
    
    # Convert pandas DataFrame to a Ray Dataset
    # This allows Ray to efficiently shard and distribute the data to workers
    dataset = ray.data.from_pandas(df)
except FileNotFoundError:
    print("ERROR: Data not found. Please run 'bash setup.sh' first.")
    exit()

# --- 2. Configure the Distributed XGBoost Trainer ---
print("Configuring the distributed trainer...")

# Define the model's parameters
xgboost_params = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "error"],
    "tree_method": "hist", # Efficient histogram-based algorithm
    "random_state": 42
}

# Configure the trainer
trainer = XGBoostTrainer(
    # This is the training function that will be executed on each worker
    scaling_config=ScalingConfig(
        # Number of parallel workers to use for training
        num_workers=4,
        # Use CPUs for training. Set to True for GPUs if available.
        use_gpu=False,
    ),
    label_column="isFraud",
    params=xgboost_params,
    datasets={"train": dataset},
    # Log results to MLflow
    run_config=ray.air.RunConfig(
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri="file:./mlruns",
                experiment_name="IEEE Fraud Detection - Distributed",
                save_artifact=True,
            )
        ]
    ),
)

# --- 3. Run Distributed Training ---
print("Starting distributed training with 4 workers...")
result = trainer.fit()

# --- 4. Print Results ---
accuracy = 1 - result.metrics["train-error"]
print(f"✅ Distributed training complete.")
print(f"   Final Accuracy: {accuracy:.4f}")
print(f"   MLflow Run Name: {result.metrics['mlflow_run_name']}")
print("\nRun 'mlflow ui' to see the results.")
# import os
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import mlflow
# from ray import tune, train
# import ray

# from ray.air.integrations.mlflow import MLflowLoggerCallback

# # --- 1. Load and Prepare Data ---
# print("Loading and preparing data...")
# try:
#     df = pd.read_csv("data/train_transaction.csv", nrows=50000)
#     df = df.select_dtypes(include='number').fillna(0)
#     X = df.drop("isFraud", axis=1)
#     y = df["isFraud"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# except FileNotFoundError:
#     print("ERROR: Data not found. Please run 'bash setup.sh' first.")
#     exit()

# # --- START OF FIX ---
# # Initialize Ray and put the large data into Ray's object store
# if not ray.is_initialized():
#     ray.init(num_cpus=4) # Adjust the number of CPUs as needed

# X_train_ref = ray.put(X_train)
# y_train_ref = ray.put(y_train)
# X_test_ref = ray.put(X_test)
# y_test_ref = ray.put(y_test)
# # --- END OF FIX ---


# # --- 2. Define the Training Function for Ray Tune ---
# def train_rf(config):
#     # --- START OF FIX ---
#     # Get the data from Ray's object store using the references
#     X_train_local = ray.get(config["X_train_ref"])
#     y_train_local = ray.get(config["y_train_ref"])
#     X_test_local = ray.get(config["X_test_ref"])
#     y_test_local = ray.get(config["y_test_ref"])
#     # --- END OF FIX ---
    
#     model = RandomForestClassifier(
#         n_estimators=config["n_estimators"],
#         max_depth=config["max_depth"],
#         random_state=42,
#         n_jobs=-1
#     )
#     model.fit(X_train_local, y_train_local)
#     y_pred = model.predict(X_test_local)
#     accuracy = accuracy_score(y_test_local, y_pred)
    
#     # Report metrics back to Ray Tune
#     train.report({"accuracy": accuracy})

# # --- 3. Configure and Run the Tuning Job ---
# print("Starting hyperparameter tuning with Ray Tune...")

# # Define the hyperparameter search space
# search_space = {
#     "n_estimators": tune.grid_search([50, 100, 150]),
#     "max_depth": tune.grid_search([5, 10, 15]),
#     # --- START OF FIX ---
#     # Pass the data references to the training function
#     "X_train_ref": X_train_ref,
#     "y_train_ref": y_train_ref,
#     "X_test_ref": X_test_ref,
#     "y_test_ref": y_test_ref,
#     # --- END OF FIX ---
# }

# # Configure the MLflow Callback
# mlflow_callback = MLflowLoggerCallback(
#     tracking_uri="file:./mlruns",
#     experiment_name="IEEE Fraud Detection - Tuning",
#     save_artifact=True
# )

# # Pass the callback to the RunConfig
# run_config = train.RunConfig(
#     name="fraud_tuning_run",
#     callbacks=[mlflow_callback],
#     stop={"training_iteration": 1}
# )

# # Start the tuning process
# tuner = tune.Tuner(
#     train_rf,
#     param_space=search_space,
#     run_config=run_config,
# )
# results = tuner.fit()

# best_trial = results.get_best_result(metric="accuracy", mode="max")
# print("\n✅ Tuning complete!")
# print(f"Best trial config: {best_trial.config}")
# print(f"Best trial accuracy: {best_trial.metrics['accuracy']:.4f}")
# print("Run 'mlflow ui' to see a detailed comparison of all tuning runs.")

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
from ray import tune, train
import ray

from ray.air.integrations.mlflow import MLflowLoggerCallback

# --- 1. Load and Prepare Data ---
print("Loading and preparing data...")
try:
    df = pd.read_csv("data/train_transaction.csv", nrows=50000)
    df = df.select_dtypes(include='number').fillna(0)
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except FileNotFoundError:
    print("ERROR: Data not found. Please run 'bash setup.sh' first.")
    exit()


# Initialize Ray and put the large data into Ray's object store
if not ray.is_initialized():
    # --- START OF FIX ---
    # Add `include_dashboard=True` to start the web UI
    ray.init(num_cpus=4, include_dashboard=True, dashboard_host='0.0.0.0')
    # --- END OF FIX ---

X_train_ref = ray.put(X_train)
y_train_ref = ray.put(y_train)
X_test_ref = ray.put(X_test)
y_test_ref = ray.put(y_test)


# --- 2. Define the Training Function for Ray Tune ---
def train_rf(config):
    # Get the data from Ray's object store using the references
    X_train_local = ray.get(config["X_train_ref"])
    y_train_local = ray.get(config["y_train_ref"])
    X_test_local = ray.get(config["X_test_ref"])
    y_test_local = ray.get(config["y_test_ref"])
    
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_local, y_train_local)
    y_pred = model.predict(X_test_local)
    accuracy = accuracy_score(y_test_local, y_pred)
    
    # Report metrics back to Ray Tune
    train.report({"accuracy": accuracy})

# --- 3. Configure and Run the Tuning Job ---
print("Starting hyperparameter tuning with Ray Tune...")

# Define the hyperparameter search space
search_space = {
    "n_estimators": tune.grid_search([50, 100, 150]),
    "max_depth": tune.grid_search([5, 10, 15]),
    # Pass the data references to the training function
    "X_train_ref": X_train_ref,
    "y_train_ref": y_train_ref,
    "X_test_ref": X_test_ref,
    "y_test_ref": y_test_ref,
}

# Configure the MLflow Callback
mlflow_callback = MLflowLoggerCallback(
    tracking_uri="file:./mlruns",
    experiment_name="IEEE Fraud Detection - Tuning",
    save_artifact=True
)

# Pass the callback to the RunConfig
run_config = train.RunConfig(
    name="fraud_tuning_run",
    callbacks=[mlflow_callback],
    stop={"training_iteration": 1}
)

# Start the tuning process
tuner = tune.Tuner(
    train_rf,
    param_space=search_space,
    run_config=run_config,
)
results = tuner.fit()

best_trial = results.get_best_result(metric="accuracy", mode="max")
print("\n✅ Tuning complete!")
print(f"Best trial config: {best_trial.config}")
print(f"Best trial accuracy: {best_trial.metrics['accuracy']:.4f}")
print("Run 'mlflow ui' to see a detailed comparison of all tuning runs.")
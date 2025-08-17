#!/usr/bin/env python3
"""
Simplified ML training script with MLflow integration
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_data(data_path: str) -> tuple:
    """Load and prepare fraud detection data"""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Basic preprocessing for demo
    # Remove columns with too many nulls
    df = df.dropna(thresh=len(df) * 0.5, axis=1)
    
    # Fill remaining nulls
    df = df.fillna(0)
    
    # Assume 'isFraud' is target column, rest are features
    if 'isFraud' in df.columns:
        y = df['isFraud']
        X = df.drop('isFraud', axis=1)
    else:
        # Create synthetic target for demo
        y = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
        X = df.select_dtypes(include=[np.number])
    
    logger.info(f"Dataset shape: {X.shape}, Target distribution: {y.value_counts()}")
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                hyperparams: Dict[str, Any]) -> RandomForestClassifier:
    """Train model with given hyperparameters"""
    logger.info(f"Training with hyperparams: {hyperparams}")
    
    model = RandomForestClassifier(
        n_estimators=hyperparams.get('n_estimators', 100),
        max_depth=hyperparams.get('max_depth', 10),
        min_samples_split=hyperparams.get('min_samples_split', 2),
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, 
                  y_test: pd.Series) -> Dict[str, float]:
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    logger.info(f"Model metrics: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument('--data', required=True, help='Path to training data')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--experiment_name', default='fraud_detection')
    
    args = parser.parse_args()
    
    # Set MLflow tracking URI (will use local if not set)
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5050')
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment(args.experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(args.experiment_name).experiment_id
    
    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id):
        # Log parameters
        hyperparams = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'min_samples_split': args.min_samples_split
        }
        mlflow.log_params(hyperparams)
        
        # Load and prepare data
        X_train, X_test, y_train, y_test = load_and_prepare_data(args.data)
        
        # Train model
        model = train_model(X_train, y_train, hyperparams)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        model_path = f"/app/models/fraud_model_{mlflow.active_run().info.run_id}.joblib"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        import joblib
        joblib.dump(model, model_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
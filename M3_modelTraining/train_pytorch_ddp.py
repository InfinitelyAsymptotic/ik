import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import ray
from ray.air.config import ScalingConfig
from ray.train.torch import TorchTrainer
from ray import train

def train_func(config: dict):
    from ray.air import session
    """Training function for a single worker."""
    # --- 1. Get Worker's Data Shard ---
    # Ray automatically shards the dataset for each worker
    train_dataset_shard = session.get_dataset_shard("train")
    
    # --- 2. Define Model and Optimizer ---
    input_size = config.get("input_size")
    model = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )
    # Prepare the model for DDP
    model = train.torch.prepare_model(model)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
    
    # --- 3. Create DataLoader for the Shard ---
    # Convert Ray Dataset shard to a PyTorch DataLoader
    dataloader = train_dataset_shard.iter_torch_batches(batch_size=config.get("batch_size"), dtypes=torch.float32)

    # --- 4. Training Loop ---
    model.train()
    for epoch in range(config.get("epochs")):
        for batch in dataloader:
            # Handle batch as dict: stack features, extract labels
            feature_keys = [k for k in batch.keys() if k != 'isFraud']
            features = torch.stack([batch[k] for k in feature_keys], dim=1)
            labels = batch['isFraud'].unsqueeze(1)
            # ...existing code...
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Report metrics back to Ray Train
        session.report({"loss": loss.item()})


# --- Main Script ---
if __name__ == "__main__":
    print("Loading data for DDP...")
    try:
        df = pd.read_csv("data/train_transaction.csv", nrows=50000)
        df = df.select_dtypes(include='number').fillna(0)
        # Add the label back as the last column for easier processing in train_func
        X = df.drop("isFraud", axis=1)
        y = df["isFraud"]
        data_with_labels = X.assign(isFraud=y)
        dataset = ray.data.from_pandas(data_with_labels)
    except FileNotFoundError:
        print("ERROR: Data not found. Please run 'bash setup.sh' first.")
        exit()

    print("Starting DDP training with PyTorch...")
    
    # Configuration for the training function
    train_config = {
        "lr": 1e-3,
        "batch_size": 128,
        "epochs": 3,
        "input_size": len(X.columns)
    }

    # Configure Ray TorchTrainer for DDP
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_config,
        scaling_config=ScalingConfig(num_workers=4, use_gpu=False),
        datasets={"train": dataset}
    )

    result = trainer.fit()
    print("✅ DDP training complete!")
    print(f"Final reported loss: {result.metrics['loss']:.4f}")
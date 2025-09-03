import torch
import torch.nn as nn
import pandas as pd
import ray

# --- 1. Define the two halves of the model ---
class Part1(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
    
    def forward(self, x):
        x = self.relu1(self.layer1(x))
        return self.layer2(x)

class Part2(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu2(x)
        x = self.sigmoid(self.layer3(x))
        return x

# --- 2. Define Ray Actors for each model part ---
# An Actor is a stateful worker process in Ray.
@ray.remote
class ModelPart:
    def __init__(self, model_part_class, **kwargs):
        self.model = model_part_class(**kwargs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def forward(self, x):
        return self.model(x)
    
    def backward(self, grads):
        # We manually perform the backward pass
        # If grads is None, or activations is not scalar, provide correct shape
        if grads is None:
            grads = torch.ones_like(self.activations)
        self.activations.backward(grads)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_input_grads(self):
        return self.input_tensor.grad
    
    def zero_grad(self):
        self.optimizer.zero_grad()

    def train_forward(self, x):
        self.zero_grad()
        self.input_tensor = x.clone().requires_grad_(True)
        self.activations = self.model(self.input_tensor)
        return self.activations


# --- Main Script ---
if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init(num_cpus=2) # Need at least 2 CPUs for 2 model parts

    print("Loading data for Model Parallelism...")
    try:
        df = pd.read_csv("data/train_transaction.csv", nrows=1000) # Smaller sample for MP demo
        df = df.select_dtypes(include='number').fillna(0)
        X = torch.tensor(df.drop("isFraud", axis=1).values, dtype=torch.float32)
        y = torch.tensor(df["isFraud"].values, dtype=torch.float32).unsqueeze(1)
    except FileNotFoundError:
        print("ERROR: Data not found. Please run 'bash setup.sh' first.")
        exit()

    print("Starting Model Parallelism training demo...")

    # --- 3. Instantiate model parts on different workers ---
    worker1 = ModelPart.remote(Part1, input_size=X.shape[1])
    worker2 = ModelPart.remote(Part2)
    
    criterion = nn.BCELoss()

    # --- 4. Manual Training Loop ---
    for epoch in range(5):
        # --- FORWARD PASS ---
        # Pass data through the first model part on worker 1
        activations_ref = worker1.train_forward.remote(X)
        activations = ray.get(activations_ref)
        
        # Pass the intermediate results to the second model part on worker 2
        outputs_ref = worker2.train_forward.remote(activations)
        outputs = ray.get(outputs_ref)

        # Calculate loss on the main process
        loss = criterion(outputs, y)
        
        # --- BACKWARD PASS ---
        # Manually compute initial gradient for the backward pass
        loss.backward()
        output_grads = outputs.grad.clone()

        # Backward pass on worker 2
        worker2.backward.remote(output_grads)
        
        # Get gradients from worker 2 to pass to worker 1
        input_grads_ref = worker2.get_input_grads.remote()
        input_grads = ray.get(input_grads_ref)
        
        # Backward pass on worker 1
        worker1.backward.remote(input_grads)
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    print("✅ Model Parallelism training demo complete!")
    ray.shutdown()
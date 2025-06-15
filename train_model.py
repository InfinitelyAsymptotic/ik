# Part 2: Train a Simple CNN Classifier on CIFAR-10
# File: train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from typing import Tuple, Dict

# Create directories
os.makedirs('saved_models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification"""
    
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_cifar10_data(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Load and preprocess CIFAR-10 dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader

def train_model(model: nn.Module, trainloader: DataLoader, 
               num_epochs: int = 5) -> Dict[str, list]:
    """Train the CNN model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    train_accuracies = []
    
    print(f"Training on {device}")
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        epoch_acc = 100 * correct / total
        train_accuracies.append(epoch_acc)
        train_losses.append(running_loss)
        print(f'Epoch {epoch + 1} Training Accuracy: {epoch_acc:.2f}%')
    
    return {'losses': train_losses, 'accuracies': train_accuracies}

def evaluate_model(model: nn.Module, testloader: DataLoader) -> float:
    """Evaluate model on test set"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def save_model_and_metadata(model: nn.Module, accuracy: float, 
                           class_names: list) -> None:
    """Save model and metadata for deployment"""
    # Save PyTorch model
    torch.save(model.state_dict(), 'saved_models/model.pth')
    
    # Save model metadata
    metadata = {
        'model_type': 'SimpleCNN',
        'num_classes': 10,
        'class_names': class_names,
        'test_accuracy': accuracy,
        'input_shape': (3, 32, 32),
        'preprocessing': {
            'normalize_mean': [0.5, 0.5, 0.5],
            'normalize_std': [0.5, 0.5, 0.5]
        }
    }
    
    joblib.dump(metadata, 'saved_models/model_metadata.joblib')
    print("Model and metadata saved successfully!")

def plot_training_progress(train_history: Dict[str, list]) -> None:
    """Plot training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_history['losses'])
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    ax2.plot(train_history['accuracies'])
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('plots/training_progress.png')
    plt.show()

if __name__ == "__main__":
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("🚀 Starting CIFAR-10 CNN Training...")
    
    # Load data
    print("📦 Loading CIFAR-10 dataset...")
    trainloader, testloader = load_cifar10_data(batch_size=64)
    
    # Initialize model
    model = SimpleCNN(num_classes=10)
    print(f"🧠 Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print("🏋️ Training model...")
    train_history = train_model(model, trainloader, num_epochs=5)
    
    # Evaluate model
    print("📊 Evaluating model...")
    test_accuracy = evaluate_model(model, testloader)
    
    # Save model and metadata
    save_model_and_metadata(model, test_accuracy, class_names)
    
    # Plot results
    plot_training_progress(train_history)
    
    print("✅ Training complete! Model ready for deployment.")

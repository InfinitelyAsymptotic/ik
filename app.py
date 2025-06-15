# Part 3: FastAPI Inference Server
# File: app.py

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import joblib
import numpy as np
import base64
import io
import uvicorn
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CIFAR-10 Image Classifier",
    description="A simple CNN-based image classifier for CIFAR-10 dataset",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    image_base64: str

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]

# Same CNN architecture as training
class SimpleCNN(nn.Module):
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

# Global variables for model and metadata
model = None
metadata = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model() -> None:
    """Load the trained model and metadata"""
    global model, metadata
    
    try:
        # Load metadata
        metadata = joblib.load('saved_models/model_metadata.joblib')
        logger.info(f"Loaded metadata: {metadata}")
        
        # Initialize and load model
        model = SimpleCNN(num_classes=metadata['num_classes'])
        model.load_state_dict(torch.load('saved_models/model.pth', map_location=device))
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for inference"""
    # Resize to 32x32 (CIFAR-10 size)
    image = image.resize((32, 32))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply same preprocessing as training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            metadata['preprocessing']['normalize_mean'],
            metadata['preprocessing']['normalize_std']
        )
    ])
    
    # Add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.to(device)

def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        return image
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Loading model...")
    load_model()
    logger.info("Model loaded successfully!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "CIFAR-10 Image Classifier API", "status": "healthy"}

@app.get("/ping")
async def ping():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if metadata is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": metadata['model_type'],
        "num_classes": metadata['num_classes'],
        "class_names": metadata['class_names'],
        "test_accuracy": metadata['test_accuracy'],
        "input_shape": metadata['input_shape']
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict image class"""
    if model is None or metadata is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        logger.info("Processing prediction request")
        
        # Decode image
        image = decode_base64_image(request.image_base64)
        
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Convert to numpy for easier handling
            probabilities_np = probabilities.cpu().numpy()[0]
            predicted_class = metadata['class_names'][predicted_idx.item()]
            confidence_score = confidence.item()
        
        # Create probability dictionary
        all_probabilities = {
            class_name: float(prob) 
            for class_name, prob in zip(metadata['class_names'], probabilities_np)
        }
        
        logger.info(f"Prediction: {predicted_class} (confidence: {confidence_score:.3f})")
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence_score,
            all_probabilities=all_probabilities
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    """Predict image class from uploaded file"""
    if model is None or metadata is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read and process uploaded file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            probabilities_np = probabilities.cpu().numpy()[0]
            predicted_class = metadata['class_names'][predicted_idx.item()]
            confidence_score = confidence.item()
        
        all_probabilities = {
            class_name: float(prob) 
            for class_name, prob in zip(metadata['class_names'], probabilities_np)
        }
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence_score,
            all_probabilities=all_probabilities
        )
        
    except Exception as e:
        logger.error(f"File prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

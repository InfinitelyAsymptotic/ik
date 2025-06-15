# Test Client for the Deployed Model
# File: test_client.py

import requests
import base64
import json
from PIL import Image
import io
import numpy as np
from typing import Dict, Any
import time

class ModelClient:
    """Client to interact with deployed ML model"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    def ping(self) -> Dict[str, Any]:
        """Test if the API is healthy"""
        try:
            response = requests.get(f"{self.base_url}/ping")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            response = requests.get(f"{self.base_url}/model-info")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_from_file(self, image_path: str) -> Dict[str, Any]:
        """Predict using image file"""
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Send prediction request
            payload = {"image_base64": image_b64}
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {"error": str(e)}
    
    def predict_from_array(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Predict using numpy array"""
        try:
            # Convert numpy array to PIL Image
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)
            
            image = Image.fromarray(image_array)
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Send prediction request
            payload = {"image_base64": image_b64}
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {"error": str(e)}

def test_with_cifar10_sample():
    """Test with a CIFAR-10 sample"""
    import torchvision
    import torchvision.transforms as transforms
    
    # Load CIFAR-10 test data
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Get a sample image
    image, true_label = testset[0]
    image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    return image_np, true_label

def benchmark_api(client: ModelClient, num_requests: int = 10) -> Dict[str, float]:
    """Benchmark API performance"""
    # Create a test image
    test_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    
    times = []
    successful_requests = 0
    
    print(f"📊 Benchmarking API with {num_requests} requests...")
    
    for i in range(num_requests):
        start_time = time.time()
        result = client.predict_from_array(test_image)
        end_time = time.time()
        
        if "error" not in result:
            times.append(end_time - start_time)
            successful_requests += 1
            print(f"Request {i+1}/{num_requests}: {end_time - start_time:.3f}s ✅")
        else:
            print(f"Request {i+1}/{num_requests}: Failed ❌ - {result['error']}")
    
    if times:
        return {
            "avg_response_time": np.mean(times),
            "min_response_time": np.min(times),
            "max_response_time": np.max(times),
            "success_rate": successful_requests / num_requests,
            "total_requests": num_requests
        }
    else:
        return {"error": "All requests failed"}

def main():
    # Initialize client - change URL for EC2 deployment
    print("🔧 Testing ML Model API")
    
    # For local testing
    client = ModelClient("http://localhost:8000")
    
    # For EC2 testing (uncomment and replace with your EC2 IP)
    # client = ModelClient("http://your-ec2-ip:8000")
    
    print("\n1️⃣ Testing API Health...")
    health = client.ping()
    print(f"Health check: {health}")
    
    print("\n2️⃣ Getting Model Info...")
    info = client.get_model_info()
    print(f"Model info: {json.dumps(info, indent=2)}")
    
    print("\n3️⃣ Testing Prediction with CIFAR-10 Sample...")
    try:
        image_array, true_label = test_with_cifar10_sample()
        result = client.predict_from_array(image_array)
        
        if "error" not in result:
            print(f"Predicted: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"True label: {true_label}")
            print("Top 3 predictions:")
            sorted_probs = sorted(
                result['all_probabilities'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for class_name, prob in sorted_probs[:3]:
                print(f"  {class_name}: {prob:.3f}")
        else:
            print(f"Prediction failed: {result['error']}")
            
    except Exception as e:
        print(f"CIFAR-10 test failed: {str(e)}")
    
    print("\n4️⃣ Performance Benchmark...")
    benchmark_results = benchmark_api(client, num_requests=5)
    if "error" not in benchmark_results:
        print(f"Average response time: {benchmark_results['avg_response_time']:.3f}s")
        print(f"Success rate: {benchmark_results['success_rate']:.1%}")
    else:
        print(f"Benchmark failed: {benchmark_results['error']}")

if __name__ == "__main__":
    main()

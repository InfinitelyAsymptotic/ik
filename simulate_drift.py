# Part 5: Model Drift Simulation
# File: simulate_drift.py

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from typing import List, Dict, Tuple
import requests
import base64
import io
from PIL import Image
import json
import time
from datetime import datetime, timedelta

class DriftSimulator:
    """Simulate model drift scenarios"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip('/')
        self.cifar10_classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.performance_log = []
    
    def predict_image(self, image_array: np.ndarray) -> Dict:
        """Send image to API for prediction"""
        try:
            # Convert numpy array to PIL Image
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)
            
            image = Image.fromarray(image_array)
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Send prediction request
            payload = {"image_base64": image_b64}
            response = requests.post(
                f"{self.api_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {"error": str(e)}
    
    def load_cifar10_test_data(self, num_samples: int = 1000) -> Tuple[List, List]:
        """Load CIFAR-10 test data"""
        transform = transforms.Compose([transforms.ToTensor()])
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        
        images, labels = [], []
        for i in range(min(num_samples, len(testset))):
            image, label = testset[i]
            image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            images.append(image_np)
            labels.append(label)
        
        return images, labels
    
    def load_cifar100_test_data(self, num_samples: int = 1000) -> Tuple[List, List]:
        """Load CIFAR-100 test data (for drift simulation)"""
        transform = transforms.Compose([transforms.ToTensor()])
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform
        )
        
        images, labels = [], []
        for i in range(min(num_samples, len(testset))):
            image, label = testset[i]
            image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            images.append(image_np)
            labels.append(label)  # These won't match CIFAR-10 classes
        
        return images, labels
    
    def add_noise_to_images(self, images: List[np.ndarray], 
                           noise_level: float = 0.3) -> List[np.ndarray]:
        """Add noise to images to simulate data quality degradation"""
        noisy_images = []
        for img in images:
            noise = np.random.normal(0, noise_level * 255, img.shape)
            noisy_img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
            noisy_images.append(noisy_img)
        return noisy_images
    
    def evaluate_batch(self, images: List[np.ndarray], 
                      true_labels: List[int], 
                      batch_name: str) -> Dict:
        """Evaluate a batch of images and log performance"""
        print(f"🔍 Evaluating {batch_name} with {len(images)} images...")
        
        predictions = []
        confidences = []
        successful_predictions = 0
        
        start_time = time.time()
        
        for i, (image, true_label) in enumerate(zip(images, true_labels)):
            result = self.predict_image(image)
            
            if "error" not in result:
                pred_class = result['predicted_class']
                confidence = result['confidence']
                
                # Convert prediction to CIFAR-10 class index
                pred_idx = self.cifar10_classes.index(pred_class) if pred_class in self.cifar10_classes else -1
                predictions.append(pred_idx)
                confidences.append(confidence)
                successful_predictions += 1
            else:
                predictions.append(-1)  # Failed prediction
                confidences.append(0.0)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(images)} images...")
        
        end_time = time.time()
        
        # Calculate metrics
        correct_predictions = sum(1 for pred, true in zip(predictions, true_labels) 
                                if pred == true and pred != -1)
        
        accuracy = correct_predictions / len(images) if images else 0
        avg_confidence = np.mean(confidences) if confidences else 0
        success_rate = successful_predictions / len(images) if images else 0
        
        metrics = {
            'batch_name': batch_name,
            'timestamp': datetime.now().isoformat(),
            'total_images': len(images),
            'successful_predictions': successful_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'success_rate': success_rate,
            'processing_time': end_time - start_time,
            'predictions': predictions,
            'true_labels': true_labels,
            'confidences': confidences
        }
        
        self.performance_log.append(metrics)
        
        print(f"📊 {batch_name} Results:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Avg Confidence: {avg_confidence:.3f}")
        print(f"  Success Rate: {success_rate:.3f}")
        print(f"  Processing Time: {end_time - start_time:.2f}s")
        
        return metrics
    
    def plot_drift_analysis(self) -> None:
        """Plot drift analysis results"""
        if not self.performance_log:
            print("No performance data to plot!")
            return
        
        # Create subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy over time
        batch_names = [log['batch_name'] for log in self.performance_log]
        accuracies = [log['accuracy'] for log in self.performance_log]
        confidences = [log['avg_confidence'] for log in self.performance_log]
        success_rates = [log['success_rate'] for log in self.performance_log]
        
        ax1.bar(batch_names, accuracies, alpha=0.7, color='skyblue')
        ax1.set_title('Model Accuracy Across Different Data Sources')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Average confidence
        ax2.bar(batch_names, confidences, alpha=0.7, color='lightcoral')
        ax2.set_title('Average Prediction Confidence')
        ax2.set_ylabel('Confidence')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Success rate
        ax3.bar(batch_names, success_rates, alpha=0.7, color='lightgreen')
        ax3.set_title('API Success Rate')
        ax3.set_ylabel('Success Rate')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        
        # Confidence distribution for latest evaluation
        if self.performance_log:
            latest_confidences = self.performance_log[-1]['confidences']
            ax4.hist(latest_confidences, bins=20, alpha=0.7, color='orange')
            ax4.set_title(f'Confidence Distribution - {self.performance_log[-1]["batch_name"]}')
            ax4.set_xlabel('Confidence')
            ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('plots/drift_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_drift_report(self) -> str:
        """Generate a comprehensive drift report"""
        if not self.performance_log:
            return "No performance data available!"
        
        report = []
        report.append("🚨 MODEL DRIFT ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary table
        report.append("📊 PERFORMANCE SUMMARY")
        report.append("-" * 30)
        for log in self.performance_log:
            report.append(f"Dataset: {log['batch_name']}")
            report.append(f"  Accuracy: {log['accuracy']:.3f}")
            report.append(f"  Confidence: {log['avg_confidence']:.3f}")
            report.append(f"  Success Rate: {log['success_rate']:.3f}")
            report.append("")
        
        # Drift detection
        report.append("🔍 DRIFT DETECTION")
        report.append("-" * 20)
        
        baseline_accuracy = self.performance_log[0]['accuracy'] if self.performance_log else 0
        
        for i, log in enumerate(self.performance_log[1:], 1):
            accuracy_drop = baseline_accuracy - log['accuracy']
            if accuracy_drop > 0.1:  # More than 10% drop
                report.append(f"⚠️  SIGNIFICANT DRIFT DETECTED in {log['batch_name']}")
                report.append(f"   Accuracy dropped by {accuracy_drop:.3f} ({accuracy_drop/baseline_accuracy:.1%})")
            elif accuracy_drop > 0.05:  # More than 5% drop
                report.append(f"⚡ MODERATE DRIFT in {log['batch_name']}")
                report.append(f"   Accuracy dropped by {accuracy_drop:.3f} ({accuracy_drop/baseline_accuracy:.1%})")
            else:
                report.append(f"✅ NO SIGNIFICANT DRIFT in {log['batch_name']}")
        
        report.append("")
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 18)
        
        if len(self.performance_log) > 1:
            latest_accuracy = self.performance_log[-1]['accuracy']
            if baseline_accuracy - latest_accuracy > 0.1:
                report.append("🔄 IMMEDIATE ACTION REQUIRED:")
                report.append("   - Retrain model with recent data")
                report.append("   - Investigate data quality issues")
                report.append("   - Consider model rollback")
            elif baseline_accuracy - latest_accuracy > 0.05:
                report.append("📈 MONITORING RECOMMENDED:")
                report.append("   - Increase monitoring frequency")
                report.append("   - Collect more labeled data")
                report.append("   - Plan retraining schedule")
            else:
                report.append("✅ CONTINUE MONITORING:")
                report.append("   - Maintain current monitoring")
                report.append("   - Regular performance checks")
        
        return "\n".join(report)

def main():
    """Main drift simulation workflow"""
    print("🚨 Starting Model Drift Simulation")
    print("=" * 50)
    
    # Initialize simulator
    simulator = DriftSimulator("http://localhost:8000")  # Change for EC2
    
    # Test API connectivity
    try:
        response = requests.get(f"{simulator.api_url}/ping", timeout=10)
        if response.status_code != 200:
            print("❌ API not accessible. Make sure the FastAPI server is running!")
            return
        print("✅ API is accessible")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return
    
    print("\n1️⃣ Evaluating on Original CIFAR-10 Data (Baseline)...")
    # Load and evaluate CIFAR-10 data (should perform well)
    cifar10_images, cifar10_labels = simulator.load_cifar10_test_data(num_samples=100)
    simulator.evaluate_batch(cifar10_images, cifar10_labels, "CIFAR-10 Baseline")
    
    print("\n2️⃣ Evaluating on Noisy CIFAR-10 Data (Quality Drift)...")
    # Add noise to simulate data quality degradation
    noisy_images = simulator.add_noise_to_images(cifar10_images, noise_level=0.2)
    simulator.evaluate_batch(noisy_images, cifar10_labels, "CIFAR-10 Noisy")
    
    print("\n3️⃣ Evaluating on Very Noisy CIFAR-10 Data (Severe Quality Drift)...")
    # Add more noise
    very_noisy_images = simulator.add_noise_to_images(cifar10_images, noise_level=0.5)
    simulator.evaluate_batch(very_noisy_images, cifar10_labels, "CIFAR-10 Very Noisy")
    
    print("\n4️⃣ Evaluating on CIFAR-100 Data (Domain Drift)...")
    # Load CIFAR-100 data (different domain, should perform poorly)
    try:
        cifar100_images, cifar100_labels = simulator.load_cifar100_test_data(num_samples=100)
        # Map CIFAR-100 labels to random CIFAR-10 labels for comparison
        fake_cifar10_labels = np.random.randint(0, 10, len(cifar100_labels))
        simulator.evaluate_batch(cifar100_images, fake_cifar10_labels.tolist(), "CIFAR-100 (Domain Drift)")
    except Exception as e:
        print(f"Failed to load CIFAR-100: {e}")
    
    print("\n5️⃣ Generating Analysis...")
    # Plot results
    simulator.plot_drift_analysis()
    
    # Generate report
    report = simulator.generate_drift_report()
    print("\n" + report)
    
    # Save report to file
    with open('drift_report.txt', 'w') as f:
        f.write(report)
    
    # Save performance log
    with open('performance_log.json', 'w') as f:
        json.dump(simulator.performance_log, f, indent=2)
    
    print("\n✅ Drift simulation complete!")
    print("📁 Files saved:")
    print("   - plots/drift_analysis.png")
    print("   - drift_report.txt") 
    print("   - performance_log.json")

if __name__ == "__main__":
    main()
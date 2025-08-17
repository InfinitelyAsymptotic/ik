#!/bin/bash
# setup.sh - Quick setup for open-source ML training

set -e

echo "🚀 Setting up Open-Source ML Training Environment"

# Create project structure
mkdir -p {data,models,logs,mlflow_data,artifacts,ray_results}

# Download sample data (if not exists)
if [ ! -f "data/sample_data.csv" ]; then
    echo "📊 Creating sample fraud detection data..."
    python3 -c "
import pandas as pd
import numpy as np

# Generate sample fraud detection data
np.random.seed(42)
n_samples = 5000

data = {
    'transaction_amount': np.random.exponential(100, n_samples),
    'account_age_days': np.random.exponential(365, n_samples),
    'num_transactions_today': np.random.poisson(5, n_samples),
    'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'merchant_risk_score': np.random.beta(2, 5, n_samples),
    'user_risk_score': np.random.beta(2, 8, n_samples),
    'isFraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
}

df = pd.DataFrame(data)
df.to_csv('data/sample_data.csv', index=False)
print('✅ Sample data created: data/sample_data.csv')
"
fi

# Build Docker image
echo "🐳 Building Docker image..."
docker build -t fraud-detector .

# Start services
echo "🚢 Starting services with Docker Compose..."
docker compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 15

# Check if MLflow is running
if curl -s http://localhost:5000 > /dev/null; then
    echo "✅ MLflow is running at http://localhost:5000"
else
    echo "❌ MLflow failed to start"
fi

# Check if Ray is running
if curl -s http://localhost:8265 > /dev/null; then
    echo "✅ Ray Dashboard is running at http://localhost:8265"
else
    echo "❌ Ray failed to start"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run training:        docker-compose exec trainer python train.py --data /app/data/sample_data.csv"
echo "2. Run hyperparameter tuning: docker-compose exec trainer python tuner.py --data /app/data/sample_data.csv"
echo "3. View MLflow UI:      http://localhost:5000"
echo "4. View Ray Dashboard:  http://localhost:8265"
echo ""
echo "To stop services:       docker-compose down"
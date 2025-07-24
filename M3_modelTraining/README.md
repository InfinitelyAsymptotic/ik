# 🧪 M3 Open-Source ML Training & Experimentation Pipeline

A minimal, production-ready machine learning training pipeline for fraud detection using open-source tools. This project demonstrates scalable model training, hyperparameter optimization, and experiment tracking without cloud vendor lock-in.

## 🎯 Features

- **🐳 Containerized Training**: docker based reproducible environments
- **📊 Experiment Tracking**: MLflow for metrics, parameters, and model versioning  
- **🔍 Hyperparameter Tuning**: Ray Tune for efficient parameter optimization
- **📈 Monitoring**: Ray Dashboard for distributed training monitoring
- **🚀 Easy Deployment**: Single command setup with Docker Compose
- **💰 Cost Effective**: No cloud service dependencies

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │    │   MLflow UI     │    │  Ray Dashboard  │
│   (CSV/DB)      │    │   :5000         │    │   :8265         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Trainer   │  │   MLflow    │  │      Ray Cluster        │ │
│  │ Container   │  │   Server    │  │    (Head + Workers)     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose installed
- 4GB+ RAM available
- 2GB+ disk space

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd fraud-detection-ml
chmod +x setup.sh
./setup.sh
```

### 2. Train Your First Model

```bash
# Basic training
docker compose exec trainer python src/train.py --data /app/data/sample_data.csv

# Training with custom parameters
docker compose exec trainer python src/train.py \
    --data /app/data/sample_data.csv \
    --n_estimators 200 \
    --max_depth 15 \
    --experiment_name "my_experiment"
```

### 3. Run Hyperparameter Tuning

```bash
# Quick tuning (10 trials)
docker compose exec trainer python src/tuner.py \
    --data /app/data/sample_data.csv \
    --num_samples 10

# Extensive tuning (50 trials)
docker compose exec trainer python src/tuner.py \
    --data /app/data/sample_data.csv \
    --num_samples 50 \
    --max_epochs 20
```

### 4. Monitor and Analyze

- **MLflow UI**: http://localhost:5050
  - View experiment runs
  - Compare model metrics
  - Download model artifacts

- **Ray Dashboard**: http://localhost:8265
  - Monitor distributed training
  - View resource utilization
  - Track tuning progress

## 📊 Using Your Own Data

### Option 1: Replace Sample Data

```bash
# Copy your CSV to the data directory
cp /path/to/your/fraud_data.csv data/

# Train with your data
docker compose exec trainer python src/train.py --data /app/data/fraud_data.csv
```

### Option 2: Use Kaggle Dataset

```bash
# Download IEEE-CIS Fraud Detection dataset
./scripts/download_data.sh

# Train with Kaggle data
docker compose exec trainer python src/train.py --data /app/data/kaggle_fraud.csv
```

## 🔧 Configuration

### Model Configuration

Edit `config/model_config.yaml`:

```yaml
model:
  type: "random_forest"
  random_state: 42
  
hyperparameters:
  n_estimators: 100
  max_depth: 10
  min_samples_split: 2
  min_samples_leaf: 1

training:
  test_size: 0.2
  cv_folds: 5
  
tuning:
  n_trials: 20
  timeout: 3600  # 1 hour
```

### Environment Variables

Copy `.env.example` to `.env` and modify:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=fraud_detection

# Ray Configuration  
RAY_DASHBOARD_HOST=0.0.0.0
RAY_DASHBOARD_PORT=8265

# Training Configuration
TRAIN_DATA_PATH=/app/data/sample_data.csv
MODEL_OUTPUT_PATH=/app/models
```

## 🧪 Advanced Usage

### Distributed Training

Scale up with additional Ray workers:

```bash
# Scale to 3 worker nodes
docker compose up --scale ray-worker=3

# Or use external Ray cluster
export RAY_ADDRESS="ray://your-cluster:10001"
python src/train.py --data data/large_dataset.csv
```

### Custom Models

Add new models in `src/models/`:

```python
# src/models/xgboost_model.py
from xgboost import XGBClassifier
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
```

### API Endpoint

Deploy trained models as REST API:

```bash
# Start model serving
docker compose exec trainer python src/serve.py --model-path /app/models/best_model.pkl

# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [100, 30, 5, 0, 0.3, 0.2]}'
```

## 📈 Monitoring & Logging

### MLflow Tracking

```python
import mlflow

# Log custom metrics
mlflow.log_metric("custom_score", 0.85)
mlflow.log_param("preprocessing", "standard_scaler")
mlflow.log_artifact("feature_importance.png")

# Register model
mlflow.sklearn.log_model(model, "fraud_detector")
```

### Ray Tune Integration

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# Custom objective function
def objective(config):
    model = RandomForestClassifier(**config)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    
    tune.report(accuracy=score)

# Run tuning
tuner = tune.Tuner(
    objective,
    param_space={
        "n_estimators": tune.randint(50, 200),
        "max_depth": tune.randint(3, 20)
    }
)
```

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
docker compose exec trainer python -m pytest tests/

# Integration tests
docker compose exec trainer python -m pytest tests/integration/

# Coverage report
docker compose exec trainer python -m pytest --cov=src tests/
```

## 🐛 Troubleshooting

### Common Issues

**MLflow UI not accessible**:
```bash
# Check if container is running
docker compose ps

# Check logs
docker compose logs mlflow
```

**Ray cluster connection failed**:
```bash
# Restart Ray services
docker compose restart ray-head ray-worker

# Check Ray status
docker compose exec ray-head ray status
```

**Out of memory during training**:
```bash
# Reduce dataset size
export SAMPLE_SIZE=10000
python src/train.py --data data/sample_data.csv --sample-size $SAMPLE_SIZE

# Or increase Docker memory limit
# Docker Desktop → Settings → Resources → Advanced → Memory
```

**Training stuck/slow**:
```bash
# Monitor resource usage
docker stats

# Check training logs
docker compose logs -f trainer
```

## 🔄 CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Run tests
      run: |
        docker compose up -d
        docker compose exec -T trainer python -m pytest tests/
        
  train:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    - name: Train model
      run: |
        docker compose up -d
        docker compose exec -T trainer python src/train.py --data data/sample_data.csv
```

## 🚀 Production Deployment

### Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker compose.prod.yml fraud-detection

# Scale services
docker service scale fraud-detection_trainer=3
```

### Kubernetes

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-detector
  template:
    metadata:
      labels:
        app: fraud-detector
    spec:
      containers:
      - name: trainer
        image: fraud-detector:latest
        ports:
        - containerPort: 8000
```

## 📚 Resources

- **MLflow Documentation**: https://mlflow.org/docs/latest/
- **Ray Tune Guide**: https://docs.ray.io/en/latest/tune/
- **Docker Best Practices**: https://docs.docker.com/develop/best-practices/
- **Fraud Detection Papers**: https://paperswithcode.com/task/fraud-detection

# Project Structure

```
fraud-detection-ml/
├── README.md
├── setup.sh
├── docker compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── train.py
│   ├── tuner.py
│   ├── data_generator.py
│   └── utils.py
│
├── config/
│   ├── model_config.yaml
│   └── logging_config.yaml
│
├── data/
│   ├── .gitkeep
│   └── sample_data.csv (generated)
│
├── models/
│   ├── .gitkeep
│   └── (saved models will go here)
│
├── logs/
│   ├── .gitkeep
│   └── (training logs will go here)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_analysis.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_train.py
│   └── test_data_generator.py
│
├── artifacts/
│   ├── .gitkeep
│   └── (MLflow artifacts will go here)
│
├── mlflow_data/
│   ├── .gitkeep
│   └── (MLflow database will go here)
│
├── ray_results/
│   ├── .gitkeep
│   └── (Ray Tune results will go here)
│
└── scripts/
    ├── download_data.sh
    ├── run_training.sh
    └── run_tuning.sh
```

## Directory Descriptions

- **src/**: Core application code
- **config/**: Configuration files for models and logging
- **data/**: Raw and processed datasets
- **models/**: Saved model artifacts
- **logs/**: Training and application logs
- **notebooks/**: Jupyter notebooks for analysis
- **tests/**: Unit and integration tests
- **artifacts/**: MLflow artifacts storage
- **mlflow_data/**: MLflow tracking database
- **ray_results/**: Ray Tune experiment results
- **scripts/**: Utility scripts for common tasks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- IEEE-CIS Fraud Detection Competition for the dataset inspiration
- MLflow team for excellent experiment tracking tools
- Ray team for distributed computing framework
- Docker community for containerization best practices

---

**Happy Machine Learning! 🚀**

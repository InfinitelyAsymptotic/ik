# 🧪 M3 Open-Source ML Training & Experimentation Pipeline

A minimal, production-ready machine learning training pipeline for fraud detection using open-source tools. This project demonstrates scalable model training, hyperparameter optimization, and experiment tracking without cloud vendor lock-in.

## 🎯 Features

- **🐳 Containerized Training**: Docker based reproducible environments
- **📊 Experiment Tracking**: MLflow for metrics, parameters, and model versioning  
- **🔍 Hyperparameter Tuning**: Ray Tune for efficient parameter optimization
- **📈 Monitoring**: Ray Dashboard for distributed training monitoring
- **🚀 Easy Deployment**: Single command setup with Docker Compose
- **💰 Cost Effective**: No cloud service dependencies
- **🎮 GPU Support**: NVIDIA GPU acceleration for faster training
- **📦 Multiple Data Sources**: Support for sample data and Kaggle datasets

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

- Docker Desktop 4.x+ (with Docker Compose v2)
- Python 3.8+
- 4GB+ RAM available
- 2GB+ disk space
- NVIDIA GPU (optional, for accelerated training)
- Kaggle account & API credentials (for downloading dataset)

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/InfinitelyAsymptotic/ik.git
cd ik/M3_modelTraining

# Initialize setup
chmod +x setup.sh
./setup.sh
```

### 2. Environment Configuration

The project uses environment variables for configuration. Create a `.env` file based on your needs:

```bash
# Create environment file
touch .env

# Add your configurations
echo "MLFLOW_TRACKING_URI=http://mlflow:5000" >> .env
echo "MLFLOW_EXPERIMENT_NAME=fraud_detection" >> .env
echo "RAY_DASHBOARD_HOST=0.0.0.0" >> .env
echo "RAY_DASHBOARD_PORT=8265" >> .env
echo "TRAIN_DATA_PATH=/app/data/sample_data.csv" >> .env
echo "MODEL_OUTPUT_PATH=/app/models" >> .env
```

Optional GPU configuration:
```bash
echo "CUDA_VISIBLE_DEVICES=0" >> .env
```

### 3. Data Setup

#### Option 1: Use Existing Sample Data
```bash
# Verify sample data exists
ls data/sample_data.csv
```

#### Option 2: Kaggle Dataset

1. Get Kaggle API credentials:
```bash
# Visit https://www.kaggle.com/settings/account
# Click "Create New API Token"
# Downloads kaggle.json
```

2. Setup credentials:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

3. Download dataset:
```bash
# Using provided script
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

The dataset structure after download:
```
data/
├── sample_data.csv
├── kaggle_fraud.csv
└── raw/
    ├── ieee-fraud-detection.zip
    ├── sample_submission.csv
    ├── test_identity.csv
    ├── test_transaction.csv
    ├── train_identity.csv
    └── train_transaction.csv
```

### 4. Training Models

#### Basic Training
```bash
# Using script
./scripts/run_training.sh

# (Optional) Direct command
docker compose exec trainer python train.py --data /app/data/sample_data.csv
```

#### Advanced Training
```bash
# Training with custom parameters
docker compose exec trainer python train.py \
    --data /app/data/kaggle_fraud.csv \
    --n_estimators 200 \
    --max_depth 15 \
    --experiment_name "custom_experiment"
```

Available parameters:
- `--data`: Path to training data CSV
- `--n_estimators`: Number of trees (default: 100)
- `--max_depth`: Maximum tree depth (default: 10)
- `--min_samples_split`: Minimum samples for split (default: 2)
- `--min_samples_leaf`: Minimum samples in leaf (default: 1)
- `--experiment_name`: MLflow experiment name

### 5. Hyperparameter Tuning

#### Quick Tuning
```bash
# Using script
./scripts/run_tuning.sh

# Direct command (10 trials)
docker compose exec trainer python tune_ray.py \
    --data /app/data/sample_data.csv \
    --num_samples 10
```

#### Advanced Tuning
```bash
# Extensive tuning (50 trials)
docker compose exec trainer python tune_ray.py \
    --data /app/data/kaggle_fraud.csv \
    --num_samples 50 \
    --max_epochs 20 \
    --cpus_per_trial 2
```

Tuning parameters:
- `--num_samples`: Number of trials (default: 10)
- `--max_epochs`: Maximum epochs per trial (default: 10)
- `--cpus_per_trial`: CPUs per trial (default: 1)
- `--gpus_per_trial`: GPUs per trial (default: 0)

### 6. Monitoring and Analysis

Access web interfaces:
- **MLflow UI**: http://localhost:5050
  - View experiment runs
  - Compare model metrics
  - Download model artifacts
  - Model versioning
- **Ray Dashboard**: http://localhost:8265
  - Monitor distributed training
  - View resource utilization
  - Track tuning progress
  - Debug execution

## 📊 Configuration

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

### Logging Configuration

Edit `config/logging_config.yaml`:

```yaml
version: 1
formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
  file:
    class: logging.FileHandler
    level: DEBUG
    formatter: standard
    filename: logs/training.log
root:
  level: INFO
  handlers: [console, file]
```

## 🐳 Docker Configuration

### Main Compose File
- `docker-compose.yml`: Primary development environment setup
- `docker.compose`: Production overrides and additional configurations

### Dockerfiles
- `Dockerfile`: Main training environment with Python, ML libraries
- `Dockerfile.mlflow`: MLflow tracking server configuration

## 🧪 Advanced Usage

### Distributed Training

Scale up with additional Ray workers:

```bash
# Scale to 3 worker nodes
docker compose up --scale ray-worker=3

# Or use external Ray cluster
export RAY_ADDRESS="ray://your-cluster:10001"
python train.py --data data/large_dataset.csv
```

### Model Management

The trained models are saved with unique identifiers in the `models/` directory:

```bash
# List saved models
ls models/
# fraud_model_1b358a55d3574046a675a55da6c69f54.joblib
# fraud_model_8fd22a7fa1fc4525a33f2090d5a14e34.joblib
# fraud_model_b5fe8ce321b143f5a55b85f0b960a221.joblib

# Load a specific model in Python
import joblib
model = joblib.load('models/fraud_model_1b358a55d3574046a675a55da6c69f54.joblib')
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

## 🐛 Troubleshooting

### Common Issues

**MLflow UI not accessible**:
```bash
# Check if container is running
docker compose ps

# Check logs
docker compose logs mlflow

# Reset MLflow database
rm -rf mlflow_data/mlflow.db
docker compose restart mlflow
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
# Reduce dataset size by sampling
head -10000 data/kaggle_fraud.csv > data/small_sample.csv
python train.py --data data/small_sample.csv

# Or increase Docker memory limit
# Docker Desktop → Settings → Resources → Advanced → Memory
```

**Training stuck/slow**:
```bash
# Monitor resource usage
docker stats

# Check training logs
docker compose logs -f trainer
tail -f logs/training.log

# Check GPU availability (if using GPU)
docker compose exec trainer nvidia-smi
```

**Kaggle dataset download fails**:
```bash
# Verify API credentials
ls ~/.kaggle/kaggle.json
cat ~/.kaggle/kaggle.json

# Test API connection
kaggle competitions list

# Manual download
kaggle competitions download -c ieee-fraud-detection
```

### GPU Issues
```bash
# Check NVIDIA driver
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check container GPU access
docker compose exec trainer nvidia-smi
```

## 🔄 CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on: [push, pull_request]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup Docker
      run: |
        docker compose up -d
        sleep 30
    - name: Train model
      run: |
        docker compose exec -T trainer python train.py --data /app/data/sample_data.csv
    - name: Save artifacts
      uses: actions/upload-artifact@v3
      with:
        name: trained-models
        path: models/
```

## 🚀 Production Deployment

### Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml fraud-detection

# Scale services
docker service scale fraud-detection_trainer=3
```

### Using Production Override

```bash
# Use production configuration
docker compose -f docker-compose.yml -f docker.compose up -d
```

## 📁 Project Structure

```
M3_modelTraining/
├── README.md
├── setup.sh
├── docker-compose.yml
├── docker.compose            # Production overrides
├── Dockerfile               # Training environment
├── Dockerfile.mlflow        # MLflow server setup
├── requirements.txt
├── train.py                 # Main training script
├── tune_ray.py              # Hyperparameter tuning
│
├── config/
│   ├── model_config.yaml    # Model parameters
│   └── logging_config.yaml  # Logging setup
│
├── data/
│   ├── sample_data.csv      # Generated sample dataset
│   ├── kaggle_fraud.csv     # Processed Kaggle data
│   └── raw/                 # Original Kaggle files
│       ├── ieee-fraud-detection.zip
│       ├── sample_submission.csv
│       ├── test_identity.csv
│       ├── test_transaction.csv
│       ├── train_identity.csv
│       └── train_transaction.csv
│
├── models/                  # Saved model artifacts
│   ├── fraud_model_1b358a55d3574046a675a55da6c69f54.joblib
│   ├── fraud_model_8fd22a7fa1fc4525a33f2090d5a14e34.joblib
│   └── fraud_model_b5fe8ce321b143f5a55b85f0b960a221.joblib
│
├── logs/                    # Training and application logs
├── artifacts/               # MLflow artifacts storage
├── mlflow_data/             # MLflow tracking database
│   └── mlflow.db
├── ray_results/             # Ray Tune experiment results
│
└── scripts/                 # Utility scripts
    ├── download_data.sh     # Kaggle data download
    ├── run_training.sh      # Training automation
    └── run_tuning.sh        # Tuning automation
```

## Directory Descriptions

- **config/**: Configuration files for models, logging, and system settings
- **data/**: Raw and processed datasets including Kaggle and sample data
  - **raw/**: Original IEEE fraud detection dataset files and zip archive
- **models/**: Saved model artifacts with unique identifiers (.joblib format)
- **logs/**: Training and application logs
- **artifacts/**: MLflow artifacts storage for experiments
- **mlflow_data/**: MLflow tracking database (mlflow.db)
- **ray_results/**: Ray Tune experiment results and hyperparameter logs
- **scripts/**: Utility scripts for data download, training, and tuning automation
- **docker.compose**: Production environment overrides for Docker Compose
- **Dockerfile.mlflow**: Specialized MLflow server container configuration

## 📚 Resources

- **MLflow Documentation**: https://mlflow.org/docs/latest/
- **Ray Tune Guide**: https://docs.ray.io/en/latest/tune/
- **Docker Best Practices**: https://docs.docker.com/develop/best-practices/
- **Fraud Detection Papers**: https://paperswithcode.com/task/fraud-detection
- **Kaggle IEEE-CIS Competition**: https://www.kaggle.com/c/ieee-fraud-detection
- **Docker Compose Reference**: https://docs.docker.com/compose/

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- IEEE-CIS Fraud Detection Competition for the dataset inspiration
- MLflow team for excellent experiment tracking tools
- Ray team for distributed computing framework
- Docker community for containerization best practices
- Kaggle community for providing high-quality datasets

---

**Happy Machine Learning! 🚀**

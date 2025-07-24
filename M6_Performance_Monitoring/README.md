# 📦 M6_Performance_Monitoring: Complete ML Pipeline with Monitoring

This module is a standalone, end-to-end ML pipeline for fraud detection including:

- Data download from Kaggle
- Data preprocessing
- Model training with MLflow experiment tracking
- Model deployment with FastAPI
- Real-time inference logging
- Data drift detection using Evidently
- Drift dashboard served via FastAPI
- Full CI/CD automation with GitHub Actions

## 📁 Project Directory Structure

```
M6_Performance_Monitoring/
├── api.py                      # FastAPI server for prediction and monitoring
├── download_data.sh            # Script to download & preprocess dataset from Kaggle
├── run_training.sh             # Wrapper to run model training
├── train.py                   # Model training with MLflow logging
├── inference_utils.py          # Preprocessor class for data handling
├── monitor.py                  # Drift detection and MLflow metric logging
├── monitoring_data_logger.py   # Logs inference requests for drift analysis
├── monitoring_utils.py         # Saves reference data snapshot for drift tests
├── requirements.txt            # Python package dependencies
├── README.md                   # This documentation file
├── .gitignore                  # Files/directories to exclude from git
└── .github/
    └── workflows/
        └── ci-cd.yml            # GitHub Actions workflow for CI/CD automation
```

## 🛠️ Setup & Usage Instructions

### 1. Clone and Setup Environment

```bash
git clone <your_repository_url>
cd M6_Performance_Monitoring

# Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Make shell scripts executable
chmod +x download_data.sh run_training.sh
```

### 2. Configure Kaggle API Credentials

The dataset is downloaded via Kaggle API, so you need to provide credentials:

1. Go to your Kaggle account settings → API → Create new API token.
2. Save the downloaded `kaggle.json` somewhere safe.
3. Run commands:

```bash
mkdir -p ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download and Preprocess Dataset

```bash
bash download_data.sh
```

This performs:
- Download IEEE-CIS Fraud Detection dataset
- Unzip and merge transaction & identity data
- Fill missing values
- Save preprocessed data as `data/kaggle_fraud.csv`

### 4. Train the Model with MLflow Tracking

```bash
bash run_training.sh
```

Or run manually with parameters:

```bash
python train.py --data data/kaggle_fraud.csv --n_estimators 100 --max_depth 10 --experiment_name fraud_monitoring_model
```

**What happens:**
- Loads and preprocesses data (encoding categorical features)
- Trains Random Forest classifier with balanced class weights
- Logs parameters and metrics (accuracy, precision, recall, F1) to MLflow
- Saves model, feature names, and preprocessor to `models/`
- Saves reference training data snapshot for drift detection at `monitoring/reference.csv`

### 5. Run the FastAPI Inference Server

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API loads the latest model and preprocessor on startup.

Provides `/predict` POST endpoint accepting JSON input:

**Example request:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 100.0,
    "dist1": 50.0,
    "card1": 1100,
    "card2": 405.0
  }'
```

**Expected JSON response:**

```json
{"isFraud": 0}
```

Every prediction input is logged to `monitoring/current.csv` for drift monitoring.

### 6. Perform Data Drift Detection

To check if input data distribution has changed ("drifted") compared to training:

```bash
python monitor.py
```

This will:
- Load `monitoring/reference.csv` (training data snapshot)
- Load `monitoring/current.csv` (recent inference data)
- Generate a detailed HTML drift report `drift_report.html` using Evidently
- Log drift metrics to MLflow for visualization and tracking

### 7. View Drift Monitoring Dashboard

Serve the latest detection report via FastAPI:

Visit: `http://localhost:8000/monitoring`

The endpoint serves the `drift_report.html` displaying interactive charts and summaries of drifted features.

### 8. Full CI/CD Automation with GitHub Actions

Every push or pull request triggers the following automated workflow defined in `.github/workflows/ci-cd.yml`:

- Checks out source code
- Installs dependencies
- Configures Kaggle credentials securely from GitHub secrets
- Downloads & preprocesses dataset
- Trains model, logs metrics and artifacts
- Runs drift detection & saves reports

No manual intervention needed once configured.

## ⚙️ Configuration Highlights

- **MLflow Tracking URI:** Local directory `mlruns/`
- **Model Storage:** `models/fraud_model_<run_id>.joblib` and related artifacts
- **Reference Data for Drift:** Saved in `monitoring/reference.csv` after training
- **Inference Logging:** Logged in `monitoring/current.csv` for drift checks
- **Preprocessing:** Handled via `inference_utils.FraudModelPreprocessor` ensuring consistent input features
- **Drift Detection:** Powered by Evidently's DataDriftPreset
- **API Endpoints:**
  - `/predict` for inference
  - `/monitoring` for drift dashboards

## 📝 Troubleshooting & Tips

- Ensure Kaggle API keys are correctly set in `~/.kaggle/kaggle.json` or GitHub secrets for CI.
- Retrain model if `models/` or `monitoring/reference.csv` missing or outdated.
- Clear or rotate `monitoring/current.csv` periodically for fresh drift checks.
- Use MLflow UI (`mlflow ui`) to explore historical experiments and drift metrics.
- Adjust `train.py` hyperparameters as needed.
- Manage virtual environment dependencies carefully (`requirements.txt`).

## 📋 Requirements

```
pandas>=1.5.0
scikit-learn>=1.1.0
mlflow>=2.0.0
joblib>=1.1.0
fastapi>=0.85.0
uvicorn>=0.18.0
evidently>=0.4.7
kaggle>=1.5.13
```

Install with:

```bash
pip install -r requirements.txt
```

## 🚀 Summary of Commands

| Command | Purpose |
|---------|---------|
| `bash download_data.sh` | Download & preprocess fraud dataset |
| `bash run_training.sh` | Train model and save artifacts |
| `uvicorn api:app --reload` | Run API server for predictions & monitoring |
| `python monitor.py` | Run drift detection & generate report |
| `curl -X POST /predict ...` | Test prediction endpoint |
| GitHub push (with secrets) | Auto-run CI/CD pipeline |
| `mlflow ui` | Launch MLflow tracking UI |

## 📚 Next Steps

- Add alerting/notifications on data drift detection (email, Slack, etc.)
- Automate drift report generation with scheduled jobs
- Extend monitor with label-based performance tracking
- Deploy model and monitoring stacks with Docker and Kubernetes
- Explore alternative drift detection packages (whylogs, deepchecks)

---

**Thank you for using M6_Performance_Monitoring!**

This module provides a solid foundation for robust ML deployment, monitoring, and CI/CD with minimal manual overhead.
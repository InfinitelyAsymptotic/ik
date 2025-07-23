# 🚦 M5_CI_CD_Automation: Fraud Detection ML Pipeline with CI/CD

A robust, open-source pipeline for fraud detection using the IEEE-CIS Kaggle dataset, complete with automated data ingestion, model training with MLflow tracking, a FastAPI deployment, and GitHub Actions-based CI/CD workflow.

---

## 📂 Project Structure

```
M5_CI_CD_Automation/
├── api.py                   # FastAPI inference server
├── download_data.sh         # Kaggle data download/preprocessing
├── run_training.sh          # Model training runner (calls train.py)

├── train.py                 # ML training with MLflow
├── requirements.txt         # Python requirements
├── README.md                # This file
├── .github/workflows/
│   └── ci-cd.yml           # GitHub Actions workflow (CI/CD)
├── data/                    # For raw/processed data
├── models/                  # For saved models
├── mlruns/                  # MLflow run storage (local)
```

---

## 🚀 Complete Setup and Usage Guide

### Step 1: Clone and Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd M5_CI_CD_Automation

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Make scripts executable
chmod +x download_data.sh run_training.sh
```

### Step 2: Configure Kaggle API Credentials

1. **Get Kaggle API Token:**
   - Go to https://www.kaggle.com/settings
   - Scroll to "API" section
   - Click "Create New API Token"
   - Download `kaggle.json`

2. **Setup Credentials:**
   ```bash
   # Create kaggle directory
   mkdir -p ~/.kaggle
   
   # Move downloaded kaggle.json to the right location
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   
   # Set proper permissions
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Verify Setup:**
   ```bash
   kaggle --version
   ```

### Step 3: Download and Preprocess Dataset

```bash
# Run the data download script
bash download_data.sh
```

**What this does:**
- Downloads IEEE-CIS Fraud Detection dataset from Kaggle
- Merges transaction and identity data
- Preprocesses and saves as `data/kaggle_fraud.csv`
- Creates necessary directories

**Expected Output:**
```
📥 Downloading IEEE-CIS Fraud Detection dataset...
🔄 Preprocessing data...
✅ Data ready!
✅ Dataset ready: data/kaggle_fraud.csv
```

### Step 4: Train the Model

```bash
# Basic training with default parameters
bash run_training.sh
```

**Or with custom parameters:**
```bash
python train.py --data data/kaggle_fraud.csv \
                --n_estimators 100 \
                --max_depth 10 \
                --experiment_name fraud_detection
```

**What this does:**
- Loads and preprocesses the fraud dataset
- Trains a Random Forest classifier
- Logs metrics and parameters to MLflow
- Saves the trained model to `models/` directory

**Expected Output:**
```
Model saved: models/fraud_model_<run_id>.joblib
```

### Step 5: Start MLflow UI (Optional)

```bash
# Start MLflow tracking UI
mlflow ui --backend-store-uri ./mlruns
```

- Open browser to http://localhost:5000
- View experiment runs, metrics, and model artifacts

### Step 6: Deploy the API Server

```bash
# Start FastAPI server
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Verify deployment:**
- API Documentation: http://localhost:8000/docs
- Health check: http://localhost:8000

### Step 7: Test Model Predictions

```bash
# Test prediction via curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 100.0,
    "dist1": 50.0,
    "card1": 1100,
    "card2": 405.0
  }'
```

**Expected Response:**
```json
{"isFraud": 0}
```

### Step 8: Setup CI/CD (GitHub Actions)

1. **Add Kaggle Secrets to GitHub:**
   - Go to your GitHub repository
   - Settings → Secrets and variables → Actions
   - Add secrets:
     - `KAGGLE_USERNAME`: Your Kaggle username
     - `KAGGLE_KEY`: Your Kaggle API key

2. **Push to trigger workflow:**
   ```bash
   git add .
   git commit -m "Initial ML pipeline setup"
   git push origin main
   ```

3. **Monitor workflow:**
   - Go to GitHub repository → Actions tab
   - Watch the CI/CD pipeline execute automatically

---

## 🛠️ Main Scripts Explained

### download_data.sh
Downloads IEEE-CIS fraud detection dataset from Kaggle and preprocesses it by merging transaction and identity data.

### train.py
Main training script that:
- Loads and cleans the dataset
- Trains Random Forest classifier
- Logs experiments with MLflow
- Saves model artifacts

### run_training.sh
Simple wrapper script to run training with predefined parameters.

### api.py
FastAPI application that:
- Loads the latest trained model
- Provides `/predict` endpoint for inference
- Handles input validation with Pydantic

### .github/workflows/ci-cd.yml
GitHub Actions workflow that:
- Triggers on push/PR
- Sets up environment
- Downloads data
- Trains model
- Ensures reproducibility

---

## 🔄 CI/CD Workflow Details

The automated pipeline performs these steps on every code change:

1. **Environment Setup**: Install dependencies
2. **Data Pipeline**: Download and preprocess Kaggle dataset
3. **Model Training**: Train model with MLflow tracking
4. **Artifact Storage**: Save models and experiment logs
5. **Quality Assurance**: Ensure reproducible results

**Workflow Triggers:**
- Push to main branch
- Pull requests
- Manual trigger

**What gets tracked:**
- Model performance metrics
- Training parameters
- Data preprocessing steps
- Model artifacts

---

## 📊 Model Performance Tracking

The pipeline tracks these metrics automatically:
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall

All metrics are logged to MLflow for experiment comparison.

---

## 🚨 Troubleshooting

### Common Issues:

1. **Kaggle API Error:**
   ```bash
   # Verify credentials
   cat ~/.kaggle/kaggle.json
   # Check permissions
   ls -la ~/.kaggle/kaggle.json
   ```

2. **Model Not Found Error:**
   ```bash
   # Ensure training completed successfully
   ls -la models/
   # Re-run training if needed
   bash run_training.sh
   ```

3. **Port Already in Use:**
   ```bash
   # Kill existing process
   pkill -f uvicorn
   # Or use different port
   uvicorn api:app --port 8001
   ```

4. **Memory Issues:**
   ```bash
   # Reduce dataset size in train.py
   # Or increase system memory
   ```

---

## 📝 Requirements

```txt
pandas>=1.5.0
scikit-learn>=1.1.0
mlflow>=2.0.0
joblib>=1.1.0
fastapi>=0.85.0
uvicorn>=0.18.0
kaggle>=1.5.13
```

---

## 💡 Advanced Usage

### Custom Model Parameters
```bash
python train.py \
  --data data/kaggle_fraud.csv \
  --n_estimators 200 \
  --max_depth 15 \
  --experiment_name custom_fraud_model
```

### Batch Predictions
```python
import requests
import json

# Multiple predictions
data = [
    {"TransactionAmt": 100.0, "dist1": 50.0, "card1": 1100, "card2": 405.0},
    {"TransactionAmt": 500.0, "dist1": 25.0, "card1": 2200, "card2": 300.0}
]

for item in data:
    response = requests.post("http://localhost:8000/predict", json=item)
    print(f"Input: {item} → Prediction: {response.json()}")
```

### Model Comparison
```bash
# Train multiple models with different parameters
python train.py --n_estimators 50 --experiment_name model_v1
python train.py --n_estimators 100 --experiment_name model_v2
python train.py --n_estimators 200 --experiment_name model_v3

# Compare in MLflow UI
mlflow ui --backend-store-uri ./mlruns
```

---

## ⚡ Quick Reference

| Command | Description |
|---------|-------------|
| `bash download_data.sh` | Download and preprocess dataset |
| `bash run_training.sh` | Train model with default settings |
| `uvicorn api:app --reload` | Start API server |
| `mlflow ui` | Open experiment tracking UI |
| `curl -X POST localhost:8000/predict -d '{...}'` | Test prediction |

### File Structure After Setup:
```
M5_CI_CD_Automation/
├── data/
│   ├── raw/                 # Original Kaggle files
│   └── kaggle_fraud.csv     # Processed dataset
├── models/
│   └── fraud_model_*.joblib # Trained models
├── mlruns/
│   └── */                   # MLflow experiment data
└── [scripts and configs]
```

---

## 🎯 Next Steps

1. **Experiment with different algorithms** (XGBoost, LightGBM)
2. **Add feature engineering** steps to improve model performance
3. **Implement model validation** and A/B testing
4. **Add monitoring** for production deployments
5. **Scale with Docker** and Kubernetes for production

---

**Happy ML engineering with automated CI/CD! 🚀**
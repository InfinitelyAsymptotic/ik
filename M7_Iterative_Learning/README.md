# 📦 M7_Iterative_Learning: Iterative, Scheduled, and Human-in-the-Loop ML (Open Source)

This module extends your full fraud ML pipeline to support true iterative retraining with:

- **Airflow** — scheduled drift checking and retrain orchestration (locally runnable, open source)
- **A/B FastAPI endpoints** — for fair model experiments and rollout
- **Human-in-the-loop labeling** — new/unusual prediction rows are sent to `label_inbox/`, you add the `isFraud` label, they become part of the next retrain
- **Continuous MLflow experiment logging**
- **Drift/notification via console** (or email, see `notify.py` for stubs)
- **All CI/CD steps via GitHub Actions**

---

## 🛠 Setup & Usage

### 1. Prepare Environment

Install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** See README from module M6 for instructions on Kaggle API setup and dataset download.

### 2. Run Data Download & Initial Training

```bash
bash download_data.sh
bash run_training.sh
```

### 3. Start FastAPI with A/B Model Endpoints

```bash
uvicorn api:app --reload
```

**Available Endpoints:**
- `/predict_a` (POST) - Model A predictions
- `/predict_b` (POST) - Model B predictions  
- `/predict_ab` (POST) - A/B testing endpoint

### 4. Human-in-the-Loop Label Workflow

1. New predictions (uncertain/drifted) automatically go to `label_inbox/to_label.csv`
2. Open and edit this file, add your ground-truth label as `isFraud`
3. Save the labeled data as `to_label_labeled.csv`
4. On retrain (Airflow or manual), these new human-labeled rows enter the model training

### 5. Schedule/Run Retrain and Drift Check

**Option A - Scheduled (Recommended):**
Start Airflow scheduler for automated retraining

**Option B - Manual:**
```bash
python monitor.py    # Check for drift
python retrain.py    # Retrain model
```

### 6. CI/CD Pipeline

Every GitHub push automatically triggers:
- Full model retrain
- Drift analysis
- Workflow updates

---

## 📊 Features

### Drift Detection & Monitoring
- Automated daily drift reports
- Console notifications (email stubs available in `notify.py`)
- Configurable drift thresholds

### A/B Testing
- Safe model rollout with A/B endpoints
- Shadow testing capabilities
- Performance comparison metrics

### Human-in-the-Loop
- Seamless integration of human feedback
- Uncertainty-based sample selection
- Audit trail for all labeled data

### Experiment Tracking
- MLflow integration for all retrains
- Model versioning and comparison
- Performance metrics logging

---

## 🎓 Quick Reference

| Step | Command/API or Folder |
|------|----------------------|
| Download dataset | `bash download_data.sh` |
| Train & save initial model | `bash run_training.sh` |
| Serve inference & A/B | `uvicorn api:app --reload` |
| Label new data for retrain | `label_inbox/to_label.csv` |
| Perform drift detection | `python monitor.py` |
| Retrain (manual or schedule) | `python retrain.py` / Airflow |
| CI/CD | Automated via GitHub Actions |

---

## 🔧 Key Files Structure

```
M7_Iterative_Learning/
├── api.py                 # FastAPI endpoints for A/B testing
├── monitor.py             # Drift detection and monitoring
├── retrain.py            # Model retraining logic
├── notify.py             # Notification utilities
├── label_inbox/          # Human-in-the-loop labeling
│   ├── to_label.csv      # Samples awaiting labels
│   └── to_label_labeled.csv  # Human-labeled samples
├── requirements.txt      # Python dependencies
├── download_data.sh      # Data download script
├── run_training.sh       # Initial training script
└── .github/workflows/    # CI/CD automation
```

---

## ✅ Advantages

- **🏠 Fully Local**: No AWS or cloud dependencies required
- **🔄 Iterative**: Continuous learning from new data
- **👥 Human-Centric**: Easy integration of human expertise
- **🔍 Transparent**: All steps auditable and open source
- **⚡ Production-Ready**: A/B testing and automated deployment

---

## 🚀 Getting Started

1. Clone this repository
2. Follow the setup steps above
3. Start with initial training
4. Launch the API for inference
5. Monitor drift and retrain as needed

**All code and steps are runnable entirely offline/local with no cloud dependencies.**
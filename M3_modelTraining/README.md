# 🧪 M3: Local ML Training & Experimentation Pipeline

A minimal, production-ready machine learning training pipeline for fraud detection using open-source tools. This project demonstrates scalable model training, hyperparameter optimization, and experiment tracking, all running locally on your machine.

## 🎯 Features

- **📊 Experiment Tracking**: MLflow for logging metrics, parameters, and model versioning.
- **🚀 Scalable Tuning**: Ray Tune for efficient, parallel hyperparameter optimization.
- **📈 Live Dashboards**: Monitor experiments with the MLflow UI and Ray Dashboard.
- **🐍 Python-Native**: Runs in a clean, isolated Python virtual environment.
- **⚡️ Fast Setup**: Get started with a single setup command.
- **📊 Kaggle Integrated**: Downloads and uses the real-world IEEE-CIS Fraud Detection dataset.

-----

## 🚀 Quick Start

### Prerequisites

Here's what you need to have installed on your computer before you begin.

- **Python 3.8+**: The programming language used for the training and tuning scripts.
- **Kaggle Account & API Credentials**: Needed to download the dataset.

-----

### 1. Project and Credentials Setup

First, get your project folder and Kaggle credentials in order.

1. **Create a Project Folder**: Create a new, empty folder on your computer to hold all the project files.

2. **Get Your Kaggle API Key**: To download the dataset, the scripts need your personal Kaggle API token.

   - **Explainer:** This token acts as a secure key, allowing the scripts to access your Kaggle account and download data on your behalf.

   **Step-by-Step Guide to Get Your Token:**

   1. **Log in to Kaggle**: Go to [https://www.kaggle.com](https://www.kaggle.com) and log in.
   2. **Go to Your Account Settings**: Click on your profile icon in the top-right and select **"Account"**.
   3. **Create New API Token**: Scroll down to the **API** section and click the **"Create New API Token"** button. This will immediately download a file named `kaggle.json`.

3. **Place Your API Key**: Move the `kaggle.json` file you just downloaded directly into your project folder.

Your project folder should now look like this:

```
your-project-folder/
├── kaggle.json
└── (all the other project files will go here)
```

-----

### 2. Environment Setup & Data Download

Next, run the automated setup script. This one-time command prepares your entire environment.

```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

**Explainer Steps:**

- `chmod +x setup.sh`: In Linux and macOS, files are not "executable" by default. This command modifies the file's permissions, granting it the permission to run.
- `./setup.sh`: This command executes the setup script, which performs several critical tasks:
  1. Creates a dedicated Python virtual environment (`.venv/`) to keep project libraries isolated.
  2. Installs all the required Python packages (like MLflow, Ray, and pandas) from `requirements.txt`.
  3. Runs the `download_data.sh` script to securely download and unzip the IEEE-CIS dataset from Kaggle into a `data/` directory.

-----

### 3. Running ML Experiments

After setup is complete, you can start training and tuning your models.

#### Activate Your Environment

First, activate the virtual environment created by the setup script. You'll need to do this every time you open a new terminal.

```bash
source .venv/bin/activate
```

- **Explainer:** This command tells your terminal to use the Python version and libraries installed specifically for this project, avoiding conflicts with other projects on your system.

### Single-Node and Basic Distributed Training

#### Basic Training (Single-Node Scikit-Learn)

This command starts the training process for a single model using default parameters.

```bash
python train.py
```

#### Hyperparameter Tuning (Parallel Trials)

This command uses Ray Tune to automatically find the best model settings from a range of possibilities.

```bash
python tune.py
```

#### Distributed Data Parallel Training (XGBoost)

This command trains a tree-based model using a data-parallel approach with **XGBoost on Ray Train**.

```bash
python train_distributed.py
```

- **Explainer:** This script uses Ray Train to distribute the dataset across multiple workers. Each worker trains an XGBoost model on its shard of the data, and the results are aggregated.

-----

### Advanced Distributed Training (PyTorch)

These scripts demonstrate advanced parallelism strategies using a PyTorch neural network. They require `torch` to be installed (`pip install -r requirements.txt`).

#### 1. Distributed Data Parallelism (DDP) with PyTorch

This script replicates the model on each worker and feeds it a different slice of data. It's the standard for speeding up training.

```bash
python train_pytorch_ddp.py
```

- **Explainer:** Shows the most common distributed training pattern where model replicas process data in parallel.

#### 2. Model Parallelism (MP) with PyTorch

This script splits a single large model across different workers. This is a demonstration of how to train models that are too big for one device's memory.

```bash
python train_pytorch_mp.py
```

- **Explainer:** Demonstrates splitting a model's layers across two workers, which then work together in a pipeline to process data.

#### 3. Fully Sharded Data Parallelism (FSDP) with PyTorch

This is the most memory-efficient technique. It shards not only the data but also the model's parameters, gradients, and optimizer state across all workers.

```bash
python train_pytorch_fsdp.py
```

- **Explainer:** Shows an advanced technique that allows for training massive models by minimizing memory redundancy on each worker.

-----

### 4. Monitoring and Analysis

While your scripts run, you can monitor the progress and analyze the results using two web-based dashboards.

#### Start the Ray Cluster and Dashboard

To have a persistent dashboard, start the Ray cluster manually in its own terminal.

1. **Start Ray**:

   ```bash
   ray start --head
   ```

   - **Explainer:** This command starts a local Ray cluster on your machine. It will print the dashboard URL to the console. **Leave this terminal running.**

2. **View the Dashboard**:

   - **Ray Dashboard**: [http://127.0.0.1:8265](http://127.0.0.1:8265)
     - **Use it to**: Monitor the distributed training and tuning jobs in real-time, see resource utilization (CPU/memory), and view logs from the Ray cluster.

3. **Stop Ray When Finished**:

   ```bash
   ray stop
   ```

   - **Explainer:** This command safely shuts down the local Ray cluster.

#### Start the MLflow UI

1. **Start the UI**:

   ```bash
   mlflow ui
   ```

   - **Explainer:** This command starts the MLflow tracking server, which reads the experiment data from your local `mlruns` directory.

2. **View the Dashboard**:

   - **MLflow UI**: [http://localhost:5000](http://localhost:5000)
     - **Use it to**: View a list of all your experiment runs, compare model metrics (like accuracy) side-by-side, and download model artifacts for any run.

-----

## Project Structure

```
.
├── .venv/
├── data/
├── mlruns/
├── ray_results/
├── scripts/
│   └── download_data.sh
├── .dockerignore
├── kaggle.json
├── README.md
├── requirements.txt
├── setup.sh
├── train.py
├── tune.py
├── train_distributed.py       # Data Parallel (XGBoost)
├── train_pytorch_ddp.py       # Data Parallel (PyTorch)
├── train_pytorch_mp.py        # Model Parallel (PyTorch)
└── train_pytorch_fsdp.py      # Fully Sharded (PyTorch)
```
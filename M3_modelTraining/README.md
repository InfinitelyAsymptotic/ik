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

-----

## 🚀 Quick Start

### Prerequisites

Here’s what you need to have installed on your computer before you begin.

  - **Docker Desktop 4.x+ (with Docker Compose v2)**: This is the core technology we use to run the project in isolated containers. It ensures that the environment is consistent and reproducible, regardless of your local machine's configuration.
  - **Python 3.8+**: The programming language used for the training and tuning scripts.
  - **4GB+ RAM available**: The processes, especially model training, can be memory-intensive.
  - **2GB+ disk space**: For storing the project files, Docker images, datasets, and trained models.
  - **NVIDIA GPU (optional, for accelerated training)**: If you have a compatible NVIDIA GPU, you can significantly speed up the model training process.
  - **Kaggle account & API credentials (for downloading dataset)**: Only needed if you want to use the larger, more realistic dataset from the Kaggle competition.

-----

### 1\. Clone and Setup

First, get the project code onto your local machine and run the initial setup.

```bash
# Clone repository
git clone https://github.com/InfinitelyAsymptotic/ik.git

# Navigate into the project directory
cd ik/M3_modelTraining

# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

**Explainer Steps:**

  * `git clone ...`: This command downloads the project's source code from the specified GitHub repository into a new folder named `ik`.
  * `cd ik/M3_modelTraining`: This changes your current location in the terminal to the project's main directory, which is where you'll run all subsequent commands.
  * `chmod +x setup.sh`: In Linux and macOS, files are not "executable" by default. This command modifies the file's permissions, granting the necessary permission to run the `setup.sh` script.
  * `./setup.sh`: This command executes the setup script. It performs initial setup tasks like creating necessary directories (`logs`, `models`, etc.) so that the application can run without errors.

-----

### 2\. Environment Configuration

Next, create a configuration file to tell the different services (like MLflow and Ray) how to communicate with each other.

```bash
# Create an empty environment file
touch .env

# Add your configurations
echo "MLFLOW_TRACKING_URI=http://mlflow:5000" >> .env
echo "MLFLOW_EXPERIMENT_NAME=fraud_detection" >> .env
echo "RAY_DASHBOARD_HOST=0.0.0.0" >> .env
echo "RAY_DASHBOARD_PORT=8265" >> .env
echo "TRAIN_DATA_PATH=/app/data/sample_data.csv" >> .env
echo "MODEL_OUTPUT_PATH=/app/models" >> .env
```

**Explainer Steps:**

  * `touch .env`: This creates an empty file named `.env`. Docker Compose automatically looks for this file to load environment variables. Using a `.env` file is a best practice for managing configuration separately from the code.
  * `echo "..." >> .env`: Each of these commands appends a line to your `.env` file.
      * `MLFLOW_TRACKING_URI`: Tells your training script where to log results. `http://mlflow:5000` points to the MLflow service running inside Docker.
      * `MLFLOW_EXPERIMENT_NAME`: Sets a name for the experiment in the MLflow UI, helping to organize your runs.
      * `RAY_DASHBOARD_HOST` and `PORT`: Configures the web address for the Ray Dashboard, which you'll use to monitor the training jobs.
      * `TRAIN_DATA_PATH`: Specifies the default dataset to use for training.
      * `MODEL_OUTPUT_PATH`: Sets the directory where the trained model files will be saved.

Optional GPU configuration:

```bash
echo "CUDA_VISIBLE_DEVICES=0" >> .env
```

  * **Explainer Step:** If you have multiple GPUs, this line tells the system to use only the first GPU (indexed at 0). If you have only one GPU, this setting is still good practice.

-----

### 3\. Data Setup

You have two options for data: use the small sample dataset included in the project or download a larger, real-world dataset from Kaggle.

#### Option 1: Use Existing Sample Data (Recommended for Quick Start)

This is the easiest way to get started. The project already includes a small sample dataset.

```bash
# Verify sample data exists
ls data/sample_data.csv
```

  * **Explainer Step:** The `ls` command lists files in a directory. This step is just to confirm that the file `sample_data.csv` is present in the `data` folder as expected.

#### Option 2: Kaggle Dataset

Follow these steps if you want to train on the larger dataset from the IEEE-CIS Fraud Detection competition on Kaggle.

1.  **Get Kaggle API credentials:**

      * **Explainer:** To download datasets programmatically, Kaggle requires a personal API token. This token acts as a secure key that allows the scripts in this project to access your Kaggle account and download data on your behalf.

    **Step-by-Step Guide to Get Your Token:**

    1.  **Log in to Kaggle**: Open your web browser and go to [https://www.kaggle.com](https://www.kaggle.com). Log in with your credentials. If you don't have an account, you will need to create one first.

    2.  **Go to Your Account Settings**: Once logged in, click on your profile picture or icon in the top-right corner and select **"Account"** from the dropdown menu. Alternatively, you can go directly to `https://www.kaggle.com/settings`.

    3.  **Find the API Section**: Scroll down the Account page until you see the **API** section.

    4.  **Create New API Token**: Click the **"Create New API Token"** button. This will immediately trigger a download of a file named `kaggle.json`.

    5.  **Save the `kaggle.json` file**: Your browser will save this file, typically to your `Downloads` folder. This file contains your unique API username and key. **Treat this file like a password and do not share it publicly.**

2.  **Setup credentials:**

    ```bash
    # Create a hidden directory for Kaggle configuration
    mkdir -p ~/.kaggle

    # Move the downloaded token to the correct directory
    mv ~/Downloads/kaggle.json ~/.kaggle/

    # Set file permissions for security
    chmod 600 ~/.kaggle/kaggle.json
    ```

      * **Explainer Steps:**
          * `mkdir -p ~/.kaggle`: The Kaggle command-line tool looks for credentials in a specific hidden folder in your home directory (`.kaggle`). This command creates that folder.
          * `mv ...`: This moves your downloaded `kaggle.json` file from `Downloads` into the `~/.kaggle` directory.
          * `chmod 600 ...`: This is an important security step. It changes the file's permissions so that only you (the owner) can read and write it, protecting your secret API key from other users on the system.

3.  **Download dataset:**

    ```bash
    # Make the download script executable
    chmod +x scripts/download_data.sh

    # Run the script to download and unzip the data
    ./scripts/download_data.sh
    ```

      * **Explainer Steps:**
          * `chmod +x ...`: Grants permission to execute the download script.
          * `./scripts/download_data.sh`: This script automates the process of using the Kaggle API to download the competition data and unzip it into the `data/raw/` directory.

-----

### 4\. Training Models

Now you are ready to train your first fraud detection model.

#### Basic Training

This command starts the training process using the default parameters.

```bash
# Using the automation script
./scripts/run_training.sh

# (Optional) The direct command that the script runs
docker compose exec trainer python train.py --data /app/data/sample_data.csv
```

  * **Explainer Steps:**
      * `./scripts/run_training.sh`: This is a convenience script that runs the full Docker command for you. It's the simplest way to start training.
      * `docker compose exec trainer ...`: This is the underlying command. Let's break it down:
          * `docker compose exec`: This is the Docker command to execute a command inside a *running* container.
          * `trainer`: This specifies the service (container) you want to run the command in, as defined in `docker-compose.yml`.
          * `python train.py`: This is the actual command to be run inside the `trainer` container. It executes the Python script that trains the model.
          * `--data /app/data/sample_data.csv`: This is an argument passed to the script, telling it which dataset to use for training.

-----

### 5\. Hyperparameter Tuning

Instead of manually guessing the best model settings, you can use Ray Tune to automatically find optimal hyperparameters.

#### Quick Tuning

This will run a small number of trials to find better hyperparameters for the model.

```bash
# Using the automation script
./scripts/run_tuning.sh

# The direct command that the script runs (for 10 trials)
docker compose exec trainer python tune_ray.py \
    --data /app/data/sample_data.csv \
    --num_samples 10
```

  * **Explainer Steps:**
      * `./scripts/run_tuning.sh`: The easiest way to start the tuning process. This script calls the more complex command for you.
      * `docker compose exec trainer python tune_ray.py ...`: This is the direct command.
          * It's similar to the training command but runs `tune_ray.py`, which is the script designed for hyperparameter optimization using Ray Tune.
          * `--num_samples 10`: This argument tells Ray Tune to try 10 different combinations of hyperparameters to find the best one.

-----

### 6\. Monitoring and Analysis

While training and tuning, you can monitor the progress and analyze the results using two web-based dashboards.

Access the web interfaces by navigating to these URLs in your browser:

  - **MLflow UI**: **http://localhost:5050**

      * **Explainer:** MLflow is for tracking your experiments. Here you can:
          * **View experiment runs**: See a list of every time you ran the training script.
          * **Compare model metrics**: Create graphs that compare the accuracy, precision, etc., of different models side-by-side.
          * **Download model artifacts**: Access and download the saved model file (`.joblib`) or any other saved files (like feature importance plots) for each run.

  - **Ray Dashboard**: **http://localhost:8265**

      * **Explainer:** Ray is the framework that manages the distributed computation. This dashboard gives you a live look into the cluster's activity. Here you can:
          * **Monitor distributed training**: See the tasks running across the Ray cluster (head and worker nodes).
          * **View resource utilization**: Check the CPU and memory usage to ensure your system is performing well.
          * **Track tuning progress**: Watch as Ray Tune launches and completes different trials in real-time.

-----

## Project Structure

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
│
├── models/                  # Saved model artifacts
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

-----

## 📚 Resources

  - **MLflow Documentation**: [https://mlflow.org/docs/latest/](https://mlflow.org/docs/latest/)
  - **Ray Tune Guide**: [https://docs.ray.io/en/latest/tune/](https://docs.ray.io/en/latest/tune/)
  - **Docker Best Practices**: [https://docs.docker.com/develop/best-practices/](https://docs.docker.com/develop/best-practices/)
  - **Kaggle IEEE-CIS Competition**: [https://www.kaggle.com/c/ieee-fraud-detection](https://www.kaggle.com/c/ieee-fraud-detection)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

-----

**Happy Machine Learning\! 🚀**
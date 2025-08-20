# 🧠 M4: Fraud Detection Model - Deployment with Docker

## 🚀 Overview

This module provides an end-to-end workflow for training a fraud detection model and deploying it as a REST API using FastAPI and Docker. You will learn how to set up your environment, train a model, and containerize the application for a consistent and scalable deployment.

-----

## 📂 Project Structure

```
.
├── api.py                   # FastAPI application for inference
├── download_data.py         # Script to download and preprocess data
├── train_model.py           # Model training script
├── requirements.txt         # Project dependencies
├── Dockerfile               # Docker configuration for the API
├── docker-compose.yml       # Docker Compose for local development
├── .gitignore               # Specifies files for Git to ignore
├── models/                  # Directory for trained model artifacts
└── data/                    # Directory for raw and processed data
```

-----

## 📌 Part 1: Initial Setup & Training

### Step 1.1: Clone Repository & Setup Environment

If you do not have the project repository already, clone it from the git repo. Then navigate to the correct directory and install the required Python packages.

```bash
# Clone the repository
git clone https://github.com/InfinitelyAsymptotic/ik.git

# Navigate into the project directory
cd ik/M4_modelDeployment

# Install all required packages
pip install -r requirements.txt
```

### Step 1.2: Kaggle API Setup

This project requires data from a Kaggle competition. You must set up Kaggle API credentials to download it.

1.  **Create Account & Join Competition:**

      * Go to the [Kaggle website](https://www.kaggle.com) and create a free account.
      * Navigate to the [IEEE-CIS Fraud Detection competition page](https://www.kaggle.com/c/ieee-fraud-detection) and accept the competition rules. The API will fail if you skip this.

2.  **Generate and Place API Token:**

      * On the Kaggle site, go to your **Account** page, scroll to the **API** section, and click **"Create New API Token"**. This will download a `kaggle.json` file.
      * Move this file to the correct location and set its permissions using the following commands:
        ```bash
        # Create the .kaggle directory if it doesn't exist
        mkdir -p ~/.kaggle

        # Move the token to that directory (assuming it's in your Downloads folder)
        mv ~/Downloads/kaggle.json ~/.kaggle/

        # Set secure permissions so only you can read it
        chmod 600 ~/.kaggle/kaggle.json
        ```

### Step 1.3: Download Data & Train Model

Now you can run the scripts to prepare the data and train the model.

```bash
# This script downloads, extracts, and preprocesses the dataset.
python download_data.py

# This script trains a RandomForest model and saves it to the models/ directory.
python train_model.py
```

The key output of this stage is the `models/fraud_model.joblib` file, which is the artifact you will deploy.

-----

## 📌 Part 2: Run and Test the API Locally with Docker

### Step 2.1: Build and Run the Container

Using Docker Compose, you can build the image and start the container with a single command.

```bash
# Build the image and start the container.
docker-compose up --build
```

The API will now be running and accessible at `http://localhost:8000`.

### Step 2.2: Test the API Endpoints

Open a **new terminal** to test the running API endpoints using `curl`.

**Prediction (`/predict`):**
This command sends a `POST` request with sample transaction data to get a fraud prediction.

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "TransactionAmt": 150.0,
    "dist1": 10.0,
    "card1": 12345.0,
    "card2": 543.0
}'
```

**Expected Response:** `{"isFraud":0}` or `{"isFraud":1}`

-----

## 📌 Part 3: Deploy to EC2 with Docker

### Step 3.1: Set up EC2 Instance

1.  **Launch an EC2 Instance:**
    * Log in to the [AWS Console](https://aws.amazon.com/console/) and navigate to EC2.
    * Click **"Launch Instance"**.
    * **Name:** Give your instance a descriptive name (e.g., "fraud-detection-api").
    * **AMI:** Select **Ubuntu Server 22.04 LTS** (free tier eligible).
    * **Instance Type:** Choose `t2.micro` (free tier) or `t2.medium` for better performance.
    * **Key Pair:** 
      - Click **"Create new key pair"** if you don't have one.
      - Name it (e.g., "fraud-detection-key").
      - Select **RSA** and **.pem** format.
      - Click **"Create key pair"** - this will download the `.pem` file to your computer.
      - **Important:** Save this file securely - you cannot download it again.

2.  **Configure Security Group:**
    * In the **Network settings** section, click **"Edit"**.
    * Add the following inbound rules:
      - **SSH (Port 22):** Source type "My IP" (automatically detects your IP).
      - **Custom TCP (Port 8000):** Source type "Anywhere" (`0.0.0.0/0`).
    * Click **"Launch Instance"**.

3.  **Set Key Permissions:**
    After downloading the `.pem` file, set secure permissions:
    ```bash
    # Navigate to where you downloaded the key (usually Downloads folder)
    cd ~/Downloads
    
    # Set secure permissions (required for SSH)
    chmod 400 your-key-name.pem
    
    # Optional: Move to a dedicated SSH keys folder
    mkdir -p ~/.ssh/aws-keys
    mv your-key-name.pem ~/.ssh/aws-keys/
    ```

### Step 3.2: Install Docker on EC2

Connect to your instance via SSH and run the following commands.

```bash
# SSH into your server
ssh -i "your-key.pem" ubuntu@your-ec2-public-ip

# Update packages and install Docker
sudo apt-get update
sudo apt-get install -y docker.io

# Add the ubuntu user to the docker group to run docker without sudo
sudo usermod -aG docker ${USER}
```

**Important:** You must **log out and log back in** for the user group change to take effect.

### Step 3.3: Deploy the Application

1.  **Copy Project to EC2:** From your **local machine**, use `scp` to copy your entire project folder.
    ```bash
    scp -r -i "your-key.pem" . ubuntu@your-ec2-public-ip:~/fraud-detection-project
    ```
2.  **Build and Run on EC2:** SSH back into your server and run the following Docker commands.
    ```bash
    # Navigate to the project folder
    cd ~/fraud-detection-project

    # Build the Docker image from the Dockerfile
    docker build -t fraud-api .

    # Run the container in detached (-d) mode, mapping port 8000 (-p)
    docker run -d --name fraud-container -p 8000:8000 fraud-api
    ```

### Step 3.4: Verify the Deployment

Your API is now live. From your local machine, test it using the server's public IP.

```bash
curl -X POST "http://your-ec2-public-ip:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "TransactionAmt": 150.0,
    "dist1": 10.0,
    "card1": 12345.0,
    "card2": 543.0
}'
```

-----

## 📌 Part 4: Troubleshooting

  * **Port 8000 Already in Use:** If you get an "address already in use" error, another service is using that port. Change the mapping in `docker-compose.yml` or the `docker run` command (e.g., `-p 8080:8000`) and access the API on the new port (`8080`).
  * **Connection Refused on EC2:** Ensure your EC2 Security Group correctly allows inbound traffic on port 8000 from `0.0.0.0/0`.
  * **Container Fails to Start:** Check the container logs for errors using the command `docker logs fraud-container`.
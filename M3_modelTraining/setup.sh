#!/bin/bash
set -e

echo "🚀 Setting up your local Python environment..."

# 1. Create a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

echo "✅ Virtual environment created."

# 2. Install dependencies
echo "🐍 Installing required libraries..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Libraries installed."

# 3. Download the dataset
echo "💾 Downloading dataset from Kaggle..."
# Make the download script executable
chmod +x download_data.sh
# Run the script
./download_data.sh

echo ""
echo "🎉 Setup complete! You can now run the training scripts."
echo "   - To train a model, run: python train.py"
echo "   - To tune a model, run:  python tune.py"
echo "   - To view results, run:  mlflow ui"
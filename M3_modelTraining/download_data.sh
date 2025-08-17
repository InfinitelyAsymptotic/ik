#!/bin/bash
set -e

# Create data directory
mkdir -p data

echo "Downloading IEEE-CIS Fraud Detection dataset..."

# Set the KAGGLE_CONFIG_DIR to the current directory to find kaggle.json
export KAGGLE_CONFIG_DIR=$(pwd)

# Download and unzip the data into the data/ directory
kaggle competitions download -c ieee-fraud-detection -p data
unzip -o data/ieee-fraud-detection.zip -d data

# Clean up the zip file
rm data/ieee-fraud-detection.zip

echo "✅ Dataset downloaded to the 'data/' folder."
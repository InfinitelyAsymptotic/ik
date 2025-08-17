# =============================================================================
# scripts/download_data.sh
# =============================================================================
#!/bin/bash
set -e

echo "📥 Downloading IEEE-CIS Fraud Detection dataset..."

# Check if kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "❌ Kaggle CLI not found. Installing..."
    pip install kaggle
fi

# Check if kaggle credentials exist
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Kaggle credentials not found."
    echo "Please download kaggle.json from https://www.kaggle.com/settings"
    echo "and place it at ~/.kaggle/kaggle.json"
    exit 1
fi

# Create data directory
mkdir -p data/raw

# Download dataset
kaggle competitions download -c ieee-fraud-detection -p data/raw/

# Extract files
cd data/raw
unzip -o ieee-fraud-detection.zip
cd ../..

# Basic preprocessing
python3 -c "
import pandas as pd
print('🔄 Preprocessing data...')

# Load main files
train_transaction = pd.read_csv('data/raw/train_transaction.csv')
train_identity = pd.read_csv('data/raw/train_identity.csv')

# Merge datasets
train_data = train_transaction.merge(train_identity, on='TransactionID', how='left')

# Basic cleaning
train_data = train_data.fillna(0)

# Save processed data
train_data.to_csv('data/kaggle_fraud.csv', index=False)
print(f'✅ Processed data saved: {train_data.shape[0]} rows, {train_data.shape[1]} columns')
"

echo "✅ Dataset ready: data/kaggle_fraud.csv"

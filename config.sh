#!/bin/bash

# === GLOBAL CONFIGURATION ===
export USER_TAG="yourname"  # change this
export BUCKET="fraud-demo-data-$USER_TAG"

export RAW_PATH="data/ieee_fraud"
export CLEANED_PATH="data/cleaned"
export CLEANED_FILE="$CLEANED_PATH/train_transaction_cleaned.csv"
export RAW_FILE="$RAW_PATH/train_transaction.csv"

export REPO_DIR="fraud_demo"
export FEAST_DATA_PATH="$REPO_DIR/data/feast-offline-store"
export FEATURE_REPO="$REPO_DIR/feature_repo"

export DVC_REMOTE="s3store"
export DVC_REMOTE_URI="s3://$BUCKET/dvcstore"

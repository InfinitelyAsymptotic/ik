#!/bin/bash
set -e
source config.sh

# System setup
sudo apt update && sudo apt install -y python3-pip unzip git python3-venv
python3 -m venv aws-env && source aws-env/bin/activate
pip install --upgrade pip setuptools
pip install 'dvc[s3]' pandas kaggle boto3 s3fs feast gitpython

# Kaggle setup
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json  # must be pre-uploaded

# Download + Upload
kaggle competitions download -c ieee-fraud-detection -p data
unzip data/ieee-fraud-detection.zip -d $RAW_PATH
aws s3 mb s3://$BUCKET || true
aws s3 cp $RAW_PATH/ s3://$BUCKET/raw/ --recursive

# DVC setup
git init && dvc init && git lfs install
dvc remote add -d $DVC_REMOTE $DVC_REMOTE_URI
dvc add $RAW_FILE
git add . .dvc .gitignore && git commit -m "Add raw dataset"
dvc push

# Preprocessing
python3 scripts/preprocess.py
dvc add $CLEANED_FILE
git add $CLEANED_FILE.dvc && git commit -m "Add cleaned dataset"
dvc push

# Feast setup
feast init $REPO_DIR
cp $CLEANED_FILE $FEAST_DATA_PATH/
cp scripts/feature_repo.py $FEATURE_REPO/example_repo.py
cd $REPO_DIR && feast apply && feast materialize-incremental $(date +%F)

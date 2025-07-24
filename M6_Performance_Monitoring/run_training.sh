#!/bin/bash

set -e
python train.py --data data/kaggle_fraud.csv --n_estimators 100 --max_depth 10 --experiment_name fraud_drift_demo

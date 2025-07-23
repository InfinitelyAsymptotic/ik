#!/bin/bash
set -e
DATA="data/kaggle_fraud.csv"
EXPERIMENT="fraud_detection"
N_EST=100
DEPTH=10
python train.py --data $DATA --experiment_name $EXPERIMENT --n_estimators $N_EST --max_depth $DEPTH
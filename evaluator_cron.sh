#!/bin/bash

# LOG_FILE="/home/jayagowtham/Documents/mlapp/evaluator_logs.log"

# exec > "$LOG_FILE" 2>&1

cd /home/jayagowtham/Documents/mlapp

source /home/jayagowtham/myenvs/.mlops/bin/activate

python3 evaluator.py

echo "Evaluation done"


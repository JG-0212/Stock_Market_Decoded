#!/bin/bash

cd /home/jayagowtham/Documents/mlapp

source /home/jayagowtham/myenvs/.mlops/bin/activate

python3 evaluator.py

# Loop through all tickers to check for DVC triggers
for ticker in $(ls evaluation); do
    score_file="evaluation/${ticker}/score.txt"
    
    if [ -f "$score_file" ]; then
        score=$(cat "$score_file")
        
        threshold=500
        
        if (( $(echo "$score < $threshold" | bc -l) )); then
            echo "Triggering DVC for $ticker (score=$score)"
            cd "${ticker}/"
            dvc repro --force
            cd ..
        else
            echo "No DVC trigger for $ticker (score=$score)"
        fi
    fi
done

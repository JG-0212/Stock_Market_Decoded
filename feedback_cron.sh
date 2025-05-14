#!/bin/bash

# LOG_FILE="/home/jayagowtham/Documents/mlapp/evaluator_logs.log"

# exec > "$LOG_FILE" 2>&1

cd /home/jayagowtham/Documents/mlapp

source /home/jayagowtham/myenvs/.mlops/bin/activate


# Loop through all tickers to check for DVC triggers
# for ticker in $(ls evaluation); do  #If any ticker fails due to rate limit, run the below line with ticker name
for ticker in AAPL; do  
    score_file="evaluation/${ticker}/score.txt"
    
    if [ -f "$score_file" ]; then
        score=$(cat "$score_file")
        
        threshold=500
        
        if (( $(echo "$score >= $threshold" | bc -l) )); then
            echo "Triggering DVC for $ticker (score=$score)"
            cd "${ticker}/"
            dvc repro --force
            cd ..

            echo "Waiting 20 seconds before next stock..."
            sleep 20
        else
            echo "No DVC trigger for $ticker (score=$score)"
        fi
    fi
done

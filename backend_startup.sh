#!/bin/bash
export MLFLOW_TRACKING_URI=http://localhost:5000

# Define models and their ports (adjust the model names as necessary)
MODEL_1_URI=$(cat ./data/AAPL/AAPL_latest.txt)
MODEL_2_URI=$(cat ./data/GOOG/GOOG_latest.txt)
MODEL_3_URI=$(cat ./data/MSFT/MSFT_latest.txt)

echo "Mlflow server at port 5000"
mlflow server --host 127.0.0.1 --port 5000 &

# Serve the models on different ports
echo "Serving AAPL $MODEL_1_URI on port 8000"
mlflow models serve --model-uri "models:/AAPLPredModel/$MODEL_1_URI" --port 8000 &

echo "Serving GOOG $MODEL_2_URI on port 8001"
mlflow models serve --model-uri "models:/GOOGPredModel/$MODEL_2_URI" --port 8001 &

echo "Serving MSFT $MODEL_3_URI on port 8002"
mlflow models serve --model-uri "models:/MSFTPredModel/$MODEL_3_URI" --port 8002 &

# Sleep indefinitely to keep the container alive
echo "Models are now being served. Press Ctrl+C to stop."
# while true; do sleep 1000; done
wait

# Copyright 2020-2024 Jordi Corbilla. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import requests
import argparse
import json
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
from mlflow.tracking import MlflowClient

from stock_prediction_class import StockPrediction
from stock_prediction_numpy import StockData
from datetime import timedelta, datetime

# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('ticker', type=str, help="The name of the stock ticker.")
    parser.add_argument('days', type=int, help="Number of days to run inference")
    
    return parser.parse_args()

def main(argv):
    
    ticker = argv.ticker
    days = argv.days 
    
    df = pd.read_csv(os.path.join(f'{ticker}',f'{ticker}_download_data.csv'))
    input = np.array(df["Close"].tail(4)) 
    input[:3] = input[1:]
    
    model_name = f"{ticker}PredModel"
    client = MlflowClient()

    version_info = client.get_latest_versions(model_name)[0]
    
        
    artifact_path = f"{ticker}_scaler.pkl"
    local_path = mlflow.artifacts.download_artifacts(
        run_id=version_info.run_id,
        artifact_path=artifact_path
        )
    
    with open(local_path, "rb") as f:
        scaler = pickle.load(f)

    preds = []
    url = "http://localhost:8000/invocations"
    while(days>0):
        pass_input = scaler.transform(input[:-1].reshape(-1,1)).reshape(1,-1).tolist()
        
        response = requests.post(url, data=json.dumps({"inputs":pass_input}), headers={"Content-Type": "application/json"})
        pred = response.json()["predictions"]

        out = scaler.inverse_transform(pred)[0][0]
        input[-1] = out
        preds.append(out)
        input[:3] = input[1:]
        
        days -= 1
    plt.plot(preds)
    plt.show()

if __name__ == '__main__':
    argv = argparser()
    main(argv)
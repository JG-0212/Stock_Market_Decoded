
import os
import pandas as pd
import numpy as np
import pickle
import json

import mlflow
import requests
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")

TICKER_PORT_MAP = {
    'AAPL' : 8000,
    'GOOG' : 8001,
    'MSFT' : 8002
}
    
def give_preds(ticker, days):

    base_path = f"data/{ticker}"
    df = pd.read_csv(os.path.join(base_path, f"{ticker}_data.csv"))
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

    pred_path = f"{ticker}_predictions.csv"
    actual_path = f"{ticker}_actual.csv"

    pred_local_path = mlflow.artifacts.download_artifacts(
        run_id=version_info.run_id,
        artifact_path=pred_path
    )
    actual_local_path = mlflow.artifacts.download_artifacts(
        run_id=version_info.run_id,
        artifact_path=actual_path
    )

    with open(pred_local_path, "rb") as f:
        val_preds = pd.read_csv(f)

    with open(actual_local_path, "rb") as f:
        val_data = pd.read_csv(f)

    preds = []
    #configurable
    url = f"http://localhost:{TICKER_PORT_MAP[ticker]}/invocations"
    
    while (days > 0):
        pass_input = scaler.transform(
            input[:-1].reshape(-1, 1)).reshape(1, -1).tolist()

        response = requests.post(url, data=json.dumps({"inputs": pass_input}), headers={
                                 "Content-Type": "application/json"})
        pred = response.json()["predictions"]

        out = scaler.inverse_transform(pred)[0][0]
        input[-1] = out
        preds.append(out)
        input[:3] = input[1:]

        days -= 1
        
    return np.array(val_data.Close[1:]), val_preds[f"{ticker}_predicted"], preds
    
def plot_preds(ticker, days):
    val_data, val_preds, future_preds = give_preds(ticker,days)
    print("plotting predictions")
    plt.figure(figsize=(14, 5))
    plt.plot(val_preds,
            color='red', label='Predicted prices')
    start = val_data.shape[0]
    ticks = range(start, start+len(future_preds))
    plt.plot(ticks, future_preds, color='blue', label="Future predictions")
    plt.plot(np.array(
        val_data, dtype='float32'), color='green', label='Actual prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Prediction')
    return plt.gcf()

def compare_preds(ticker_list, days):
    plt.figure(figsize=(14, 5))
    for ticker in ticker_list:
        _, _, future_preds = give_preds(ticker,days)
        plt.plot(future_preds,  label=ticker)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Prediction')
    return plt.gcf()


if __name__ == '__main__':
    pass
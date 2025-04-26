
from datetime import timedelta, datetime
from stock_prediction_numpy import StockData
from stock_prediction_class import StockPrediction
from mlflow.tracking import MlflowClient
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import json
import argparse
import requests
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def plot_predictions(stock_ticker, val_preds, val_data, future_preds=None):
    print("plotting predictions")
    plt.figure(figsize=(14, 5))
    plt.plot(val_preds[stock_ticker + '_predicted'],
             color='red', label='Predicted prices')
    if future_preds is not None:
        start = val_data.shape[0]-2
        ticks = range(start, start+len(future_preds))
        plt.plot(ticks, future_preds, color='blue', label="Future predictions")
    plt.plot(np.array(
        val_data.Close[2:], dtype='float32'), color='green', label='Actual prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Prediction')
    # plt.savefig(os.path.join(f'{self.stock_ticker}',f'{self.stock_ticker}_prediction.png'))
    # plt.show()
    return plt.gcf()


def give_preds_and_plots(ticker, days):

    # ticker = argv.ticker
    # days = argv.days

    df = pd.read_csv(os.path.join(f'{ticker}', f'{ticker}_download_data.csv'))
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
    url = "http://localhost:8000/invocations"
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

    return plot_predictions(ticker, val_preds, val_data, preds)


if __name__ == '__main__':
    # argv = argparser()
    give_preds_and_plots()

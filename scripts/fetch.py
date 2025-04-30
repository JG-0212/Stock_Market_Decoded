import os
from datetime import datetime

import mlflow
import pandas as pd
import yfinance as yf

from utils.helpers import get_jan_first_years_ago, read_yaml

if __name__ == '__main__':
    
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Stock Price Forecasting")
    default_start_date = get_jan_first_years_ago(8).strftime("%Y-%m-%d")

    config = read_yaml('params.yaml')
    #parameters
    ticker = config["base"]["ticker"]
    start_date = pd.to_datetime(default_start_date)
    end_date = pd.to_datetime(datetime.today())
    
    #configurable
    base_path = "../data"
    directory_path = os.path.join(base_path, ticker)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
    run = mlflow.start_run(run_name=f"{ticker}_{datetime.today().strftime('%Y-%m-%d')}")
    mlflow.log_params({
        "ticker": ticker,
        "start_date": default_start_date
    })
    
    with open(os.path.join(directory_path,f"{ticker}_run_id.txt"), "w") as f:
        f.write(run.info.run_id)
        
    data = yf.download(ticker, start=start_date, end=end_date)[['Close']]
    data = data.reset_index()
    data.to_csv(os.path.join(directory_path, f"{ticker}_data.csv"), index=False)

    
import os
import logging
from datetime import datetime

import mlflow
import pandas as pd
import yfinance as yf

from utils.helpers import get_jan_first_years_ago, read_yaml

# Logger setup with script name and formatted output
logger = logging.getLogger("fetch")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename='pipeline_logs.log',
    filemode="w"
)

if __name__ == '__main__':
    try:
        logger.info("Starting fetch.py")

        try:
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("Stock Price Forecasting")
            logger.info("MLflow tracking URI and experiment set.")
        except Exception:
            logger.exception("Failed to configure MLflow.")
        
        try:
            default_start_date = get_jan_first_years_ago(8).strftime("%Y-%m-%d")
            config = read_yaml('params.yaml')
            logger.info("Configuration loaded.")
        except Exception:
            logger.exception("Failed to read configuration")

        try:
            ticker = config["base"]["ticker"]
            start_date = pd.to_datetime(default_start_date)
            end_date = pd.to_datetime(datetime.today())
        except Exception:
            logger.exception("Failed to parse dates or ticker from config.")

        try:
            base_path = "../data"
            directory_path = os.path.join(base_path, ticker)
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                logger.info(f"Created directory: {directory_path}")
        except Exception:
            logger.exception("Failed to create or access directory path.")

        try:
            run = mlflow.start_run(run_name=f"{ticker}_{datetime.today().strftime('%Y-%m-%d')}")
            mlflow.log_params({
                "ticker": ticker,
                "start_date": default_start_date
            })
            logger.info("MLflow run started and parameters logged.")
        except Exception:
            logger.exception("Failed to start MLflow run or log parameters.")

        try:
            with open(os.path.join(directory_path, f"{ticker}_run_id.txt"), "w") as f:
                f.write(run.info.run_id)
            logger.info(f"Run ID saved for ticker: {ticker}")
        except Exception:
            logger.exception("Failed to save MLflow run ID.")

        try:
            data = yf.download(ticker, start=start_date, end=end_date)[['Close']]
            data = data.reset_index()
            data.to_csv(os.path.join(directory_path, f"{ticker}_data.csv"), index=False)
            logger.info(f"Data fetched and saved for ticker: {ticker}")
        except Exception:
            logger.exception("Failed to fetch or save stock data.")

    except Exception:
        logger.exception("An unexpected error occurred in fetch.py")

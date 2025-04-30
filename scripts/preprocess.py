# preprocess.py
import os
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import mlflow

from utils.helpers import get_jan_first_years_ago, data_statistics, read_yaml

# Configure logger
logger = logging.getLogger("preprocess")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename='pipeline_logs.log',
    filemode="a"
)

if __name__ == '__main__':
    try:
        logger.info("Starting preprocess.py")

        try:
            mlflow.set_tracking_uri("http://localhost:5000")
            logger.info("MLflow tracking URI set.")
        except Exception:
            logger.exception("Failed to set MLflow tracking URI.")

        try:
            default_start_date = get_jan_first_years_ago(8).strftime("%Y-%m-%d")
            default_val_date = get_jan_first_years_ago(2).strftime("%Y-%m-%d")
            config = read_yaml('params.yaml')
            logger.info("Config and default dates loaded.")
        except Exception:
            logger.exception("Failed to load configuration or compute default dates.")

        try:
            ticker = config["base"]["ticker"]
            start_date = pd.to_datetime(default_start_date)
            val_date = pd.to_datetime(default_val_date)
            time_steps = config["base"]["time_steps"]
        except Exception:
            logger.exception("Failed to parse config values.")

        try:
            base_path = "../data"
            directory_path = os.path.join(base_path, ticker)
            data = pd.read_csv(os.path.join(directory_path, f"{ticker}_data.csv"))
            data['Date'] = pd.to_datetime(data['Date'])
            logger.info("Data loaded and parsed.")
        except Exception:
            logger.exception("Failed to load or process data file.")

        try:
            with open(os.path.join(directory_path, f"{ticker}_run_id.txt"), "r") as f:
                run_id = f.read().strip()
            mlflow.start_run(run_id=run_id)
            mlflow.log_params({
                "val_date": default_val_date,
                "time_steps": time_steps
            })
            logger.info("MLflow run started and parameters logged.")
        except Exception:
            logger.exception("Failed to start MLflow run or log parameters.")

        try:
            training_data = data[data['Date'] < val_date].copy()
            val_data = data[data['Date'] >= val_date].copy()
            training_data = training_data.set_index('Date')
            val_data = val_data.set_index('Date')
            logger.info("Training and validation data split.")
        except Exception:
            logger.exception("Failed to split data.")

        try:
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_scaled = scaler.fit_transform(training_data.values)
            with open(os.path.join(directory_path, f"{ticker}_scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact(os.path.join(directory_path, f"{ticker}_scaler.pkl"))
            data_statistics(train_scaled)
            logger.info("Scaler saved and data statistics logged.")
        except Exception:
            logger.exception("Failed during scaling or statistics step.")

        try:
            x_train, y_train = [], []
            for i in range(time_steps, train_scaled.shape[0]):
                x_train.append(train_scaled[i - time_steps:i])
                y_train.append(train_scaled[i, 0])
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            logger.info("Training data transformed.")
        except Exception:
            logger.exception("Failed to transform training data.")

        try:
            total_data = pd.concat((training_data, val_data), axis=0)
            inputs = total_data[len(total_data) - len(val_data) - time_steps:]
            val_scaled = scaler.transform(inputs.values)
            x_val, y_val = [], []
            for i in range(time_steps, val_scaled.shape[0]):
                x_val.append(val_scaled[i - time_steps:i])
                y_val.append(val_scaled[i, 0])
            x_val = np.array(x_val)
            y_val = np.array(y_val)
            x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
            logger.info("Validation data transformed.")
        except Exception:
            logger.exception("Failed to transform validation data.")

        try:
            np.save(os.path.join(directory_path, f"{ticker}_x_train.npy"), x_train)
            np.save(os.path.join(directory_path, f"{ticker}_y_train.npy"), y_train)
            np.save(os.path.join(directory_path, f"{ticker}_x_val.npy"), x_val)
            np.save(os.path.join(directory_path, f"{ticker}_y_val.npy"), y_val)
            logger.info("Training and validation numpy arrays saved.")
        except Exception:
            logger.exception("Failed to save numpy arrays.")

        try:
            training_data.to_csv(os.path.join(directory_path, f"{ticker}_training_data.csv"))
            val_data.to_csv(os.path.join(directory_path, f"{ticker}_val_data.csv"))
            logger.info("Raw training and validation DataFrames saved.")
        except Exception:
            logger.exception("Failed to save CSVs.")

    except Exception:
        logger.exception("An unexpected error occurred in preprocess.py")

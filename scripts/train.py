# train.py
import os
import pickle
import logging

import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from utils.helpers import read_yaml
from utils.lstm_model import LongShortTermMemory
from utils.plotters import plot_histogram_data_split, plot_loss, plot_predictions

# Logger configuration
logger = logging.getLogger("train")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename='pipeline_logs.log',
    filemode="a"
)

if __name__ == '__main__':
    try:
        logger.info("Starting train.py")
        
        try:
            mlflow.set_tracking_uri("http://localhost:5000")
            config = read_yaml("params.yaml")
            logger.info("MLflow and config loaded.")
        except Exception:
            logger.exception("Failed to set MLflow URI or read config.")

        try:
            ticker = config["base"]["ticker"]
            patience = config["base"]["patience"]
            epochs = config["base"]["epochs"]
            batch_size = config["base"]["batch_size"]
            base_path = "../data"
            directory_path = os.path.join(base_path, ticker)
            logger.info(f"Ticker parameters loaded: {ticker}")
        except Exception:
            logger.exception("Failed to load config values.")

        try:
            with open(os.path.join(directory_path, f"{ticker}_run_id.txt"), "r") as f:
                run_id = f.read().strip()
            mlflow.start_run(run_id=run_id)
            mlflow.log_params({
                "patience": patience,
                "epochs": epochs,
                "batch_size": batch_size
            })
            logger.info("MLflow run started and parameters logged.")
        except Exception:
            logger.exception("Failed to start MLflow run or log parameters.")

        try:
            x_train = np.load(os.path.join(directory_path, f"{ticker}_x_train.npy"))
            y_train = np.load(os.path.join(directory_path, f"{ticker}_y_train.npy"))
            x_val = np.load(os.path.join(directory_path, f"{ticker}_x_val.npy"))
            y_val = np.load(os.path.join(directory_path, f"{ticker}_y_val.npy"))
            logger.info("Training and validation data loaded.")
        except Exception:
            logger.exception("Failed to load training/validation data arrays.")

        try:
            training_data = pd.read_csv(os.path.join(directory_path, f"{ticker}_training_data.csv"))
            val_data = pd.read_csv(os.path.join(directory_path, f"{ticker}_val_data.csv"))
            training_data['Date'] = pd.to_datetime(training_data['Date'])
            val_data['Date'] = pd.to_datetime(val_data['Date'])
            training_data.set_index('Date', inplace=True)
            val_data.set_index('Date', inplace=True)
            logger.info("Training and validation CSVs loaded and indexed.")
        except Exception:
            logger.exception("Failed to load or process training/validation CSVs.")

        try:
            plot_histogram_data_split(ticker, training_data, val_data, directory_path)
            logger.info("Data histogram plotted.")
        except Exception:
            logger.exception("Failed to plot histogram.")

        try:
            lstm = LongShortTermMemory(patience)
            model = lstm.create_model(x_train)
            model.compile(optimizer='adam', loss='mean_squared_error')
            history = model.fit(
                x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_val, y_val),
                callbacks=[lstm.get_callback()]
            )
            logger.info("Model trained successfully.")
        except Exception:
            logger.exception("Model training failed.")

        try:
            mlflow.keras.log_model(
                model, "lstm_model", registered_model_name=f"{ticker}PredModel")
            logger.info("Model logged to MLflow.")
        except Exception:
            logger.exception("Failed to log model to MLflow.")

        try:
            baseline_results = model.evaluate(x_val, y_val, verbose=2)
            mlflow.log_metric(model.metrics_names[0], baseline_results)
            logger.info(f"Validation loss: {baseline_results}")
        except Exception:
            logger.exception("Model evaluation failed.")

        try:
            plot_loss(ticker, history, directory_path)
            logger.info("Loss plot saved.")
        except Exception:
            logger.exception("Failed to plot loss.")

        try:
            with open(os.path.join(directory_path, f"{ticker}_scaler.pkl"), "rb") as f:
                scaler = pickle.load(f)

            val_predictions_baseline = model.predict(x_val)
            val_predictions_baseline = scaler.inverse_transform(val_predictions_baseline)
            val_predictions_baseline = pd.DataFrame(val_predictions_baseline)
            val_predictions_baseline.rename(columns={0: f"{ticker}_predicted"}, inplace=True)
            val_predictions_baseline = val_predictions_baseline.round(decimals=0)
            val_predictions_baseline.index = val_data.index

            val_predictions_baseline.to_csv(os.path.join(directory_path, f"{ticker}_predictions.csv"))
            val_data.to_csv(os.path.join(directory_path, f"{ticker}_actual.csv"))

            mlflow.log_artifact(os.path.join(directory_path, f"{ticker}_predictions.csv"))
            mlflow.log_artifact(os.path.join(directory_path, f"{ticker}_actual.csv"))

            logger.info("Predictions and actuals saved and logged.")
        except Exception:
            logger.exception("Failed during prediction or file saving.")

        try:
            plot_predictions(ticker, val_predictions_baseline, val_data, directory_path)
            logger.info("Prediction plot saved.")
        except Exception:
            logger.exception("Failed to plot predictions.")

        try:
            plot_base_name = f"{ticker}_"
            sub_names = ["loss.png", "price.png", "prediction.png"]
            plot_names = [os.path.join(directory_path, plot_base_name + sn) for sn in sub_names]
            for plot_name in plot_names:
                if os.path.exists(plot_name):
                    mlflow.log_artifact(plot_name)
            logger.info("Optional plots logged to MLflow.")
        except Exception:
            logger.exception("Failed to log plots to MLflow.")

        try:
            model_name = f"{ticker}PredModel"
            client = MlflowClient()
            version_info = client.get_latest_versions(model_name)[0]
            with open(f"../data/{ticker}/{ticker}_latest.txt", "w") as f:
                f.write(str(version_info.version))
            logger.info(f"Latest model version written: {version_info.version}")
        except Exception:
            logger.exception("Failed to log latest model version.")

        mlflow.end_run()
        logger.info("train.py completed successfully.")

    except Exception:
        logger.exception("Unexpected failure in train.py")

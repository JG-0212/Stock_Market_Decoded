# trains the model
# fetches the latest data from data/ and preprocesses it plus saves it
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import mlflow

from utils.helpers import read_yaml
from utils.lstm_model import LongShortTermMemory
from utils.plotters import plot_histogram_data_split, plot_loss, plot_predictions

if __name__ == '__main__':
    
    
    mlflow.set_tracking_uri("http://localhost:5000")
    config = read_yaml("params.yaml")
    #parameters
    ticker = config["base"]["ticker"]
    patience = config["base"]["patience"]
    epochs = config["base"]["epochs"]
    batch_size = config["base"]["batch_size"]
    
    #configurable
    base_path = "/home/jayagowtham/Documents/mlapp/data"
    directory_path = os.path.join(base_path, ticker)
    
    with open(os.path.join(directory_path,f"{ticker}_run_id.txt"), "r") as f:
        run_id = f.read().strip()
    mlflow.start_run(run_id=run_id)
    mlflow.log_params({
        "patience": patience,
        "epochs": epochs,
        "batch_size": batch_size
    })
    x_train = np.load(os.path.join(directory_path, f"{ticker}_x_train.npy"))
    y_train = np.load(os.path.join(directory_path, f"{ticker}_y_train.npy"))
    x_val = np.load(os.path.join(directory_path, f"{ticker}_x_val.npy"))
    y_val = np.load(os.path.join(directory_path, f"{ticker}_y_val.npy"))
    
    training_data = pd.read_csv(os.path.join(directory_path, f"{ticker}_training_data.csv"))
    val_data = pd.read_csv(os.path.join(directory_path, f"{ticker}_val_data.csv"))
    
    training_data['Date'] = pd.to_datetime(training_data['Date'])
    training_data.set_index('Date', inplace=True)

    val_data['Date'] = pd.to_datetime(val_data['Date'])
    val_data.set_index('Date', inplace=True)


    
    plot_histogram_data_split(ticker, training_data, val_data, directory_path)

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

    # Save model to MLflow
    mlflow.keras.log_model(
        model, "lstm_model", registered_model_name=f"{ticker}PredModel")

    # Log final metrics
    baseline_results = model.evaluate(x_val, y_val, verbose=2)
    print(model.metrics_names[0], ': ', baseline_results)
    mlflow.log_metric(model.metrics_names[0], baseline_results)

    # Optionally log plots
    plot_loss(ticker, history, directory_path)

    with open(os.path.join(directory_path, f"{ticker}_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
        
    val_predictions_baseline = model.predict(x_val)
    val_predictions_baseline = scaler.inverse_transform(val_predictions_baseline)
    val_predictions_baseline = pd.DataFrame(val_predictions_baseline)
    val_predictions_baseline.rename(columns={0: f"{ticker}_predicted"}, inplace=True)
    val_predictions_baseline = val_predictions_baseline.round(decimals=0)
    val_predictions_baseline.index = val_data.index
    val_predictions_baseline.to_csv(os.path.join(directory_path,f"{ticker}_predictions.csv"))
    val_data.to_csv(os.path.join(directory_path, f"{ticker}_actual.csv"))

    # Optionally log predictions file
    mlflow.log_artifact(os.path.join(directory_path, f"{ticker}_predictions.csv"))
    mlflow.log_artifact(os.path.join(directory_path, f"{ticker}_actual.csv"))

    # Log plots if saved
    plot_predictions(ticker, val_predictions_baseline, val_data, directory_path)

    plot_base_name = f"{ticker}_"
    sub_names = ["loss.png", "price.png", "prediction.png"]
    plot_names = [os.path.join(directory_path, plot_base_name+sn) for sn in sub_names]
    for plot_name in plot_names:  # Assuming you save plots with these names
        if os.path.exists(plot_name):
            mlflow.log_artifact(plot_name)

    print("Model training is finished")
    mlflow.end_run()



    
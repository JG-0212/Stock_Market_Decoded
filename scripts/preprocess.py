# fetches the latest data from data/ and preprocesses it plus saves it
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import mlflow

from utils.helpers import get_jan_first_years_ago, data_statistics

if __name__ == '__main__':
    default_start_date = get_jan_first_years_ago(8).strftime("%Y-%m-%d")
    default_val_date = get_jan_first_years_ago(2).strftime("%Y-%m-%d")
    
    #parameters
    ticker = 'GOOG'
    start_date = pd.to_datetime(default_start_date)
    val_date = pd.to_datetime(default_val_date)
    time_steps = 3
    
    #configurable
    base_path = "/home/jayagowtham/Documents/mlapp/data"
    directory_path = os.path.join(base_path, ticker)
    data = pd.read_csv(os.path.join(directory_path, f"{ticker}_data.csv"))
    data['Date'] = pd.to_datetime(data['Date'])
    
    with open(os.path.join(directory_path,f"{ticker}_run_id.txt"), "r") as f:
        run_id = f.read().strip()
    mlflow.start_run(run_id=run_id)
    
    mlflow.log_params({
        "val_date": default_val_date,
        "time_steps": time_steps
    })
    
    training_data = data[data['Date'] < val_date].copy()
    val_data = data[data['Date'] >= val_date].copy()
    
    training_data = training_data.set_index('Date')
    val_data = val_data.set_index('Date')

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(training_data)
    with open(os.path.join(directory_path,f"{ticker}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    mlflow.log_artifact(os.path.join(directory_path,f"{ticker}_scaler.pkl"))
    data_statistics(train_scaled)

    # Training Data Transformation
    x_train = []
    y_train = []
    for i in range(time_steps, train_scaled.shape[0]):
        x_train.append(train_scaled[i - time_steps:i])
        y_train.append(train_scaled[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    total_data = pd.concat((training_data, val_data), axis=0)
    inputs = total_data[len(total_data) - len(val_data) - time_steps:]
    val_scaled = scaler.transform(inputs)

    # Validation Data Transformation
    x_val = []
    y_val = []
    for i in range(time_steps, val_scaled.shape[0]):
        x_val.append(val_scaled[i - time_steps:i])
        y_val.append(val_scaled[i, 0])

    x_val, y_val = np.array(x_val), np.array(y_val)
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    
    np.save(os.path.join(directory_path, f"{ticker}_x_train.npy"), x_train)
    np.save(os.path.join(directory_path, f"{ticker}_y_train.npy"), y_train)
    np.save(os.path.join(directory_path, f"{ticker}_x_val.npy"), x_val)
    np.save(os.path.join(directory_path, f"{ticker}_y_val.npy"), y_val)

    # Save raw DataFrames
    training_data.to_csv(os.path.join(directory_path, f"{ticker}_training_data.csv"))
    val_data.to_csv(os.path.join(directory_path, f"{ticker}_val_data.csv"))


    
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
import secrets
import pandas as pd
import argparse
from datetime import datetime

import mlflow

from stock_prediction_class import StockPrediction
from stock_prediction_lstm import LongShortTermMemory
from stock_prediction_numpy import StockData
from stock_prediction_plotter import Plotter



def train_LSTM_network(stock):
   with mlflow.start_run(run_name=stock.get_ticker()+'_'+datetime.today().strftime("%Y-%m-%d")):
        mlflow.log_params({
            "ticker": stock.get_ticker(),
            "start_date": stock.get_start_date().strftime("%Y-%m-%d"),
            "validation_date": stock.get_validation_date().strftime("%Y-%m-%d"),
            "epochs": stock.get_epochs(),
            "batch_size": stock.get_batch_size(),
            "time_steps": stock.get_time_steps()
        })

        data = StockData(stock)
        plotter = Plotter(data.get_stock_short_name(), data.get_stock_currency(), stock.get_ticker())
        (x_train, y_train), (x_test, y_test), (training_data, test_data) = data.download_transform_to_numpy(stock.get_time_steps())
        plotter.plot_histogram_data_split(training_data, test_data, stock.get_validation_date())

        lstm = LongShortTermMemory()
        model = lstm.create_model(x_train)
        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(
            x_train, y_train,
            epochs=stock.get_epochs(),
            batch_size=stock.get_batch_size(),
            validation_data=(x_test, y_test),
            callbacks=[lstm.get_callback()]
        )

        # print("saving weights")
        # model.save(os.path.join(stock.get_project_folder(), 'model_weights.h5'))

        # Save model to MLflow
        mlflow.keras.log_model(model, "lstm_model")

        # Log final metrics
        baseline_results = model.evaluate(x_test, y_test, verbose=2)
        print(model.metrics_names[0], ': ', baseline_results)
        mlflow.log_metric(model.metrics_names[0], baseline_results)

        # Optionally log plots
        plotter.plot_loss(history)
        
        test_predictions_baseline = model.predict(x_test)
        test_predictions_baseline = data.get_min_max().inverse_transform(test_predictions_baseline)
        test_predictions_baseline = pd.DataFrame(test_predictions_baseline)
        test_predictions_baseline.rename(columns={0: stock.get_ticker() + '_predicted'}, inplace=True)
        test_predictions_baseline = test_predictions_baseline.round(decimals=0)
        test_predictions_baseline.index = test_data.index
        test_predictions_baseline.to_csv(f'{stock.get_ticker()}_predictions.csv')

        # Optionally log predictions file
        mlflow.log_artifact(f'{stock.get_ticker()}_predictions.csv')

        # Log plots if saved
        plotter.project_plot_predictions(test_predictions_baseline, test_data)
        
        plot_base_name = f'{stock.get_ticker()}_'
        sub_names = ["loss.png", "price.png", "prediction.png"]
        plot_names = [plot_base_name+sn for sn in sub_names]
        for plot_name in plot_names:  # Assuming you save plots with these names
            if os.path.exists(plot_name):
                mlflow.log_artifact(plot_name)

        print("prediction is finished")

def get_jan_first_years_ago(years_back):
    today = datetime.today()
    return datetime(today.year - years_back, 1, 1)


if __name__ == '__main__':
    
    mlflow.set_experiment("Stock Price Forecasting")
    mlflow.set_tracking_uri("http://localhost:5000")


    default_start_date = get_jan_first_years_ago(6).strftime("%Y-%m-%d")
    default_validation_date = get_jan_first_years_ago(2).strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description=("parsing arguments"))
    parser.add_argument("-ticker", default="GOOG")
    parser.add_argument("-start_date", default=default_start_date)
    parser.add_argument("-validation_date", default=default_validation_date)
    parser.add_argument("-epochs", default="100")
    parser.add_argument("-batch_size", default="10")
    parser.add_argument("-time_steps", default="3")
    
    args = parser.parse_args()
    
    STOCK_TICKER = args.ticker
    
    directory_path = STOCK_TICKER
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    STOCK_START_DATE = pd.to_datetime(args.start_date)
    STOCK_VALIDATION_DATE = pd.to_datetime(args.validation_date)
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    TIME_STEPS = int(args.time_steps)
    TODAY_RUN = datetime.today().strftime("%Y%m%d")
    print('Ticker: ' + STOCK_TICKER)
    print('Start Date: ' + STOCK_START_DATE.strftime("%Y-%m-%d"))
    print('Validation Date: ' + STOCK_VALIDATION_DATE.strftime("%Y-%m-%d"))

    stock_prediction = StockPrediction(STOCK_TICKER, 
                                       STOCK_START_DATE, 
                                       STOCK_VALIDATION_DATE, 
                                       EPOCHS,
                                       TIME_STEPS,
                                       BATCH_SIZE)
    # Execute Deep Learning model
    train_LSTM_network(stock_prediction)

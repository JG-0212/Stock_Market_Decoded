import os
import matplotlib.pyplot as plt

import yfinance as yf



def plot_histogram_data_split(ticker, training_data, val_data, directory_path):
    print("Plotting train data plots")
    plt.figure(figsize=(12, 5))
    plt.plot(training_data['Close'], label="Training Data", color='green')
    plt.plot(val_data['Close'], label="Validation Data", color='red')
    plt.ylabel(f'Price)')
    plt.xlabel("Date")
    plt.title(f"{ticker} Price Split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(directory_path, f"{ticker}_price_split.png"))
    plt.close()

    numeric_data = training_data.select_dtypes(include='number')

    _ = numeric_data.hist(figsize=(12, 8), bins=30, edgecolor='black')
    plt.suptitle(f"{ticker} - Feature Distribution", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
    plt.savefig(os.path.join(directory_path, f"{ticker}_feature_histograms.png"))
    plt.close()



def plot_loss(ticker, history, directory_path):
    print("Plotting loss curve...")
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get('loss', []),
             label='Training Loss', color='blue', linewidth=2)
    plt.plot(history.history.get('val_loss', []), label='Validation Loss',
             color='orange', linestyle='--', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{ticker} - Training vs Validation Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(directory_path, f"{ticker}_loss.png")
    plt.savefig(output_path)
    plt.close()


def plot_predictions(ticker, val_preds, val_data, directory_path, future_preds=None):
    print("Plotting predictions...")
    plt.figure(figsize=(14, 6))
    plt.plot(val_data['Close'].values, color='green', label='Actual Price')
    plt.plot(val_preds[f"{ticker}_predicted"].values, color='red', label='Validation Prediction')
    if future_preds is not None:
        start = len(val_data)
        future_ticks = range(start, start + len(future_preds))
        plt.plot(future_ticks, future_preds, color='blue', linestyle='--', label='Future Prediction')
    plt.xlabel('Days')
    plt.ylabel(f'Price')
    plt.title(f'{ticker} - Price Predictions')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(directory_path, f"{ticker}_predictions.png")
    plt.savefig(output_path)
    plt.close()

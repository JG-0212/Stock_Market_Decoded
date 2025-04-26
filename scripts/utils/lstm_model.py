import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import Input, Dropout, Dense, LSTM
from tensorflow.keras import Sequential
import tensorflow as tf



class LongShortTermMemory:
    def __init__(self, patience):
        self.patience = patience

    def get_callback(self):
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, mode='min', verbose=1)
        return callback

    def create_model(self, x_train):
        model = Sequential()
        model.add(Input(shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(units=50))
        model.add(Dropout(0.5))
        model.add(Dense(units=1))
        model.summary()

        return model
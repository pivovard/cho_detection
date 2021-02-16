import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import load_log
import load_data
import wg


ID = 591
cho_l = 'Carbohydrate intake'
ist_l = 'Interstitial glucose'

MAX_EPOCHS = 2
WINDOW_WIDTH_1H = 12 #1 hour window
WINDOW_WIDTH_24H = 288 #24 hour window

#load_log.load_log(patientID=ID)
#df = load_data.load_data(patientID=ID, verbose=True, graphs=True)
df = load_data.load_data(patientID=ID, from_file=True, verbose=False, graphs=False)


headers = ['Interstitial glucose', 'Carbohydrate intake', 'hour', 'weekday']
labels = 'Carbohydrate intake'

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

#test WindowGenerator
wg.test(df)

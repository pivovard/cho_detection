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
#wg.test(df)

data = pd.DataFrame()
for i, h in enumerate(['Interstitial glucose', 'Carbohydrate intake', 'hour', 'weekday']):
    data[h] = df[h]

ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=df['Carbohydrate intake'],
        sequence_length=WINDOW_WIDTH_24H,
        sequence_stride=1,
        shuffle=False,
        batch_size=32)

lstm_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
lstm_model.fit(ds, epochs=MAX_EPOCHS, callbacks=[early_stopping])

ds_test = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=WINDOW_WIDTH_24H,
        sequence_stride=1,
        shuffle=False,
        batch_size=32)

predictions = lstm_model(next(iter(ds_test)))


plt.figure(figsize=(12, 8))
plt.scatter(df['datetime'][0:WINDOW_WIDTH_24H], predictions[0],
            marker='o', edgecolors='k', label='Labels', c='#2ca02c', s=10)
plt.scatter(df['datetime'][0:WINDOW_WIDTH_24H], df['Carbohydrate intake'][0:WINDOW_WIDTH_24H],
            marker='^', edgecolors='k', label='Predictions',
            c='#ff7f0e', s=10)

plt.show()
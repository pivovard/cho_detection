import matplotlib.pyplot as plt

import load_log
import load_data
import cho_detection as cho
import pa_detection as pa
from WindowGenerator import WindowGenerator
import nn
import utils

# wrong 540
# good  570 575
ID = 575
IDs=[540,544,552,559,563,570,575,584,588,591,596]

# Parse log file to csv file
# load_log.load_log(patientID=ID)
# load_log.load_log_all(IDs)

# Load data from csv file
# df = load_data.load_data(ID, label='Interstitial glucose', fill_missing='',
#                          smooth='savgol', derivation='difference', norm='',
#                          verbose=True, graphs=True, analyze=False)
# Load modified data from file
df = load_data.load_data(ID, label='Interstitial glucose', from_file=True, verbose=True, graphs=False, analyze=False)
# Load multiple csv files
# df = load_data.load_data_all(IDs, from_file=True, fill_missing='', smooth='savgol', derivation='difference', norm='')

plt.show()

## IST prediction
# headers = [utils.ist_l, utils.inr_l, utils.inb_l, 'hour', 'weekday', 'der1', 'der2', 'der3']
# window = WindowGenerator(df=df, headers=headers, label_columns=['Interstitial glucose'],
#                               input_width=utils.WINDOW_WIDTH_1H*3, label_width=6, shift=6)
# model = nn.feedback(window)
# nn.predict(model, window)

## CHO prediction
headers = [utils.ist_l, 'weekday', 'd1', 'd2', 'd3']
# cho.lda_window(df, ['Interstitial glucose'], 24, 'window')
# cho.lda(df, headers, 'multiple values')

headers = ['ist', 'd1', 'minute_n']
# cho.lstm(df, headers,'cho2', 'GRU', epochs=100, patientID=ID)
# cho.lstm_test(df[30*utils.WINDOW_WIDTH_24H:32*utils.WINDOW_WIDTH_24H], headers, 'Carbohydrate intake', 15, path=f'model/{ID}_keras_model.h5')
# cho.lstm_test(df[:2*utils.WINDOW_WIDTH_24H], headers, 'Carbohydrate intake', 15, path=f'model/{ID}_keras_model.h5')
# cho.lstm_test(df, headers, 'Carbohydrate intake', 15, path=f'model/{ID}_keras_model.h5')

# 575 NECHAT JAKO UKAZKOVY!!!
act = cho.threshold(df[30*utils.WINDOW_WIDTH_24H:32*utils.WINDOW_WIDTH_24H])
# act = cho.threshold(df)
# utils.evaluate(df['cho_b'], act, treshold=3)
# utils.evaluate(df['cho_b'], act, treshold=5.5)

# df['act'] = act
# headers = ['ist', 'd1', 'minute_n', 'act']
# cho.lstm(df, headers,'cho2', 'GRU', epochs=100, patientID=ID)

## PA prediction
headers=['Heartbeat', 'Steps', 'Electrodermal activity', 'Skin temperature']
headers=['Acceleration']
# pa.get_pa(ID, df, headers)
# pa.predict_pa(ID, headers)
# headers = ['Heartbeat', 'Steps', 'Skin temperature']
# pa.lstm(df, headers,'pa', 'GRU', epochs=20, patientID=ID)
# pa.dense(patientID, headers,'pa')
# pa.lstm_test(df[30*utils.WINDOW_WIDTH_24H:32*utils.WINDOW_WIDTH_24H], headers, 'pa', 15, path=f'model/{ID}_keras_model.h5')

plt.show()
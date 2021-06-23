"""
Examples of use

@author Bc. David Pivovar
"""

import matplotlib.pyplot as plt

import alg.load_log as lg
import alg.load_data as ld
import alg.cho_detection as cho
import alg.pa_detection as pa
from alg.WindowGenerator import WindowGenerator
import alg.nn as nn
import alg.utils as utils


ID = 559
IDs=[540,544,552,563,570,575,584,591,596]


## Parse log file to csv file
# lg.load_log(patientID=ID)
# lg.load_log_all(IDs, 'testing')


## Load data from csv file
# df = ld.load_data(ID, label='Interstitial glucose', fill_missing='',
#                          smooth='savgol', derivation='difference', norm='',
#                          verbose=True, graphs=True, analyze=False)
## Load modified data from file
# df = ld.load_data_file(ID, label='Interstitial glucose', verbose=True, graphs=False, analyze=False)
## Load multiple csv files
# df = ld.load_data_all(IDs, from_file=True, fill_missing='', smooth='savgol', derivation='difference', norm='')


## IST prediction
# headers = [utils.ist_l, utils.inr_l, utils.inb_l, 'hour', 'weekday', 'der1', 'der2', 'der3']
# window = WindowGenerator(df=df, headers=headers, label_columns=['Interstitial glucose'],
#                               input_width=utils.WINDOW_WIDTH_1H*3, label_width=6, shift=6)
# model = nn.feedback(window)
# nn.predict(model, window)


## CHO prediction LDA
# headers = [utils.ist_l, 'weekday', 'd1', 'd2', 'd3']
# cho.lda_window(df, 'Interstitial glucose', 24, 'window')
# cho.lda(df, headers, 'multiple values')

## CHO prediction RNN
# headers = ['ist', 'd1', 'minute_n']
# cho.lstm(df, headers,'cho2', 'GRU', epochs=100, patientID=1)
# cho.lstm_test(df, headers, 'Carbohydrate intake', 12, path=f'model/{ID}_keras_model.h5')
# plt.show()

## load data and train RNN for all pacients
# for i, ID in enumerate(IDs):
#     df = ld.load_data(ID, label='Interstitial glucose', fill_missing='',
#                          smooth='savgol', derivation='difference', norm='',
#                          verbose=True, graphs=False, analyze=False)
#     cho.lstm(df, headers,'cho2', 'GRU', epochs=100, patientID=ID)

## CHO prediction threshold
# act = cho.threshold(df)
# utils.evaluate(df['cho_b'], act, treshold=3)
# utils.evaluate(df['cho_b'], act, treshold=5.5)

# plt.show()


## PA prediction
## Load data from csv file
# df = ld.load_data(ID, label='Heartbeat', fill_missing='',
#                          smooth='', derivation='difference', norm='',
#                          verbose=True, graphs=True, analyze=False)
## Load modified data from file
# df = ld.load_data_file(ID, label='Heartbeat', verbose=True, graphs=False, analyze=False)

# headers=['Heartbeat', 'Steps', 'Electrodermal activity', 'Skin temperature']
# headers=['Heartbeat', 'Steps']
# headers=['Acceleration']

## Get features from the given columns
# df = pa.get_features(ID, df, headers)
# pa.export_pa(df, headers, ID)

## Get features of multiple patients
# IDs=[570,575,588,591]
# for i, ID in enumerate(IDs):
#     df = ld.load_data(ID, type='testing', label='Heartbeat', fill_missing='',
#                          smooth='', derivation='difference', norm='',
#                          verbose=True, graphs=True, analyze=False)
#     pa.get_features(ID, df, headers, test='-test')

## Test machine learning algorithms
# pa.ML(ID, headers)

# plt.show()
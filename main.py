import matplotlib.pyplot as plt

import load_log
import load_data
from cho_detection import ChoDetector
from WindowGenerator import WindowGenerator
import nn
import utils

# wrong 540
# good  570 575
ID = 575
IDs=[540,544,552,559,563,567,570,575,584,588,591,596]

# Parse log file to csv file
# load_log.load_log(patientID=ID)
# load_log.load_log_all(IDs)

# Load data from csv file
# df = load_data.load_data(ID, fill_missing='', smooth='savgol', derivation='manual', norm='',
#     verbose=True, graphs=True, analyze=False)
# Load modified data from file
df = load_data.load_data(ID, from_file=True, verbose=True, graphs=False, analyze=False)
# Load multiple csv files
# df = load_data.load_data_all(IDs, from_file=True, fill_missing='', smooth='savgol', derivation='manual', norm='')

plt.show()

## IST prediction
# headers = [utils.ist_l, utils.inr_l, utils.inb_l, 'hour', 'weekday', 'der1', 'der2', 'der3']
# window = WindowGenerator(df=df, headers=headers, label_columns=['Interstitial glucose'],
#                               input_width=utils.WINDOW_WIDTH_1H*3, label_width=6, shift=6)
# model = nn.feedback(window)
# nn.predict(model, window)

## CHO prediction
cho = ChoDetector(df, True)
headers = ['Interstitial glucose', 'd1', 'hour']
# cho.lda()
# cho.lstm(headers,'cho2')
cho.lstm_test(headers, 'Carbohydrate intake', 15, 'keras_model.h5', df[30*utils.WINDOW_WIDTH_24H:32*utils.WINDOW_WIDTH_24H])
cho.lstm_test(headers, 'Carbohydrate intake', 15, 'keras_model.h5', df[:2*utils.WINDOW_WIDTH_24H])

# 575 NECHAT JAKO UKAZKOVY!!!
# act = cho.treshold_manual(df[30*utils.WINDOW_WIDTH_24H:32*utils.WINDOW_WIDTH_24H])
# act = cho.treshold_manual(df)
# cho.evaluate(df['cho_b'], act, treshold=3)

## PA prediction
# headers = ['Heartbeat', 'Steps', 'Skin temperature']
# cho.lstm(headers,'pa')
# cho.lstm_test(headers, 'pa', 15, 'cho_model.h5', df[30*utils.WINDOW_WIDTH_24H:32*utils.WINDOW_WIDTH_24H])

plt.show()
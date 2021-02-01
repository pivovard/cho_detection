import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import load_log
import load_data
import nn


ID = 591
cho_l = 'Carbohydrate intake'
ist_l = 'Interstitial glucose'
WINDOW_WIDTH_1H = 12 #1 hour window
WINDOW_WIDTH_24H = 288 #24 hour window

#load_log.load_log(patientID=ID)
#df = load_data.load_data(patientID=ID, verbose=True, graphs=True)
df = load_data.load_data(patientID=ID, from_file=True, verbose=False, graphs=False)


headers = ['Interstitial glucose', 'Carbohydrate intake', 'hour', 'weekday']

window = nn.WindowGenerator(data=df, headers=headers, input_width=WINDOW_WIDTH_24H, label_width=1, shift=0, label_columns=['Carbohydrate intake']) #shift= 0 - label last row; 1 - prediction 1 row;
print(window.train.element_spec)
    
window.plot(show=True)

plt.show()
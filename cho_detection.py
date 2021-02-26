import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import timedelta

import load_log
import load_data
from wg import WindowGenerator
import nn
import utils

ID = 591

# Parse log file to csv file
#load_log.load_log(patientID=ID)

# Load data from csv file
df = load_data.load_data(patientID=ID, verbose=True, graphs=True)
# Load modified data from file
# df = load_data.load_data(patientID=ID, from_file=True, verbose=True, graphs=False)

utils.printh('Derivations')
der = pd.DataFrame(columns=['d1', 'd2', 'd3'], dtype=float)
for i in range(len(df)-1):
    if (df.loc[i+1,'datetime']-df.loc[i,'datetime']) > timedelta(0, 0, 0, 0, 5, 0):
        der.loc[i] = [0,0,0]
        continue
    d1=(df.loc[i+1,utils.ist_l]-df.loc[i,utils.ist_l])/5
    d2=d1/5
    d3=d2/5
    der.loc[i] = [d1,d2,d3]

fig = plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.title('Derivations')
plt.plot(df['datetime'][1:], df[utils.ist_l][1:], label='ist')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(df['datetime'][1:], der['d1'], label='d1')
plt.plot(df['datetime'][1:], der['d2'], label='d2')
plt.plot(df['datetime'][1:], der['d3'], label='d3')
plt.legend()
plt.show()

# nn.single_step(df)
# nn.multi_step(df)

headers = [utils.ist_l, utils.inr_l, utils.inb_l, 'hour', 'weekday']
window = WindowGenerator(data=df, headers=headers, label_columns=['Interstitial glucose'],
                              input_width=utils.WINDOW_WIDTH_1H*3, label_width=6, shift=6)
model = nn.feedback(window)

nn.predict(model, window)
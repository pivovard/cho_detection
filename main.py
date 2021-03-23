import matplotlib.pyplot as plt

import load_log
import load_data
import cho_detection
from WindowGenerator import WindowGenerator
import nn
import nn_cho
import utils


ID = 552
headers1 = [utils.ist_l, utils.inr_l, 'hour', 'weekday', 'grad1', 'grad2', 'grad3']
headers2 = [utils.ist_l, 'grad1', 'grad2', 'grad3']
headers3 = [utils.ist_l]

# Parse log file to csv file
# load_log.load_log(patientID=ID)

# Load data from csv file
# df = load_data.load_data(patientID=ID, fill_missing=True, der=False, grad=True,
#                          verbose=True, graphs=True, analyze=True)
# Load modified data from file
df = load_data.load_data(patientID=ID, from_file=True,
                         verbose=True, graphs=False, analyze=False)

# nn.single_step(df)
# nn.multi_step(df)

# headers = [utils.ist_l, utils.inr_l, utils.inb_l, 'hour', 'weekday', 'der1', 'der2', 'der3']
# window = WindowGenerator(df=df, headers=headers, label_columns=['Interstitial glucose'],
#                               input_width=utils.WINDOW_WIDTH_1H*3, label_width=6, shift=6)
# model = nn.feedback(window)

# nn.predict(model, window)

# cho_detection.LDA(df)
cho_detection.LSTM(df)

plt.show()
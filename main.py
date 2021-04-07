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

# Parse log file to csv file
# load_log.load_log(patientID=ID)

# Load data from csv file
# df = load_data.load_data(patientID=ID, fill_missing='', smooth='savgol', derivation='manual', norm=False,
#     verbose=True, graphs=True, analyze=False)
# Load modified data from file
df = load_data.load_data(patientID=ID, from_file=True,
                         verbose=True, graphs=False, analyze=False)

# plt.show()

# nn.single_step(df)
# nn.multi_step(df)W

# headers = [utils.ist_l, utils.inr_l, utils.inb_l, 'hour', 'weekday', 'der1', 'der2', 'der3']
# window = WindowGenerator(df=df, headers=headers, label_columns=['Interstitial glucose'],
#                               input_width=utils.WINDOW_WIDTH_1H*3, label_width=6, shift=6)
# model = nn.feedback(window)

# nn.predict(model, window)

cho = ChoDetector(df, True)
# cho.lda()
# cho.lstm()
# NECHAT JAKO UKAZKOVY!!!
# cho.treshold_manual(df[30*utils.WINDOW_WIDTH_24H:32*utils.WINDOW_WIDTH_24H])
cho.treshold_manual(df[20*utils.WINDOW_WIDTH_24H:21*utils.WINDOW_WIDTH_24H])

plt.show()
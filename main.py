import matplotlib.pyplot as plt

import load_log
import load_data
from cho_detection import ChoDetector
from WindowGenerator import WindowGenerator
import nn
import utils


ID = 570

# Parse log file to csv file
# load_log.load_log(patientID=ID)

# Load data from csv file
df = load_data.load_data(patientID=ID, fill_missing='akima', norm=True, derivation='akima',
                         verbose=True, graphs=True, analyze=True)
# Load modified data from file
df = load_data.load_data(patientID=ID, from_file=True,
                         verbose=True, graphs=False, analyze=False)

plt.show()

# nn.single_step(df)
# nn.multi_step(df)

# headers = [utils.ist_l, utils.inr_l, utils.inb_l, 'hour', 'weekday', 'der1', 'der2', 'der3']
# window = WindowGenerator(df=df, headers=headers, label_columns=['Interstitial glucose'],
#                               input_width=utils.WINDOW_WIDTH_1H*3, label_width=6, shift=6)
# model = nn.feedback(window)

# nn.predict(model, window)

cho = ChoDetector(df, True)
cho.lda()

plt.show()
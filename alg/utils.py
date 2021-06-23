"""
Support functions.

@author Bc. David Pivovar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ist_l = 'Interstitial glucose'
cho_l = 'Carbohydrate intake'
phy_l = 'Physical activity'
inb_l = 'requested insulin bolus'
inr_l = 'requested insulin basal rate'


WINDOW_WIDTH_1H = 12    #1 hour window
WINDOW_WIDTH_24H = 288  #24 hour window

def print_h(text):
    print('\n' + '+' * (len(text)+6))
    print(f'++ {text} ++')
    print('+' * (len(text)+6) + '\n')

# Print iterations progress
def printProgressBar (iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 50, fill = '=', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '>' + '.' * (length - filledLength)
    print(f'\r{prefix} [{bar}] {percent}% {suffix}', end = printEnd)

def evaluate(y_label, y_pred, treshold=0, method=''):
    n = len(y_label)
    TP = np.zeros(n)
    FN = np.zeros(n)
    FP = np.zeros(n)
    cho_count = 0
    delay = 0
    w = 24 #2h
    wf = 36 #3h
    y_elements = []

    for i, y in enumerate(y_label):
        if y > 0:
            cho_count +=1
            y_elements = y_pred[i:i+w]
            if np.any(y_elements > treshold):
                TP[i] = True
                delay += 5*np.argmax(y_elements >= treshold)
            else:
                FN[i] = True
        elif y_pred[i] > treshold:
            if i >= wf:
                y_elements = y_label[i-wf:i+wf]
            else:
                y_elements = y_label[:i+wf]
            if not np.any(y_elements > 0) and np.all(FP[i-wf:i]==False):
                FP[i] = True
    
    print(method)
    print(f'CHO: {cho_count}')
    print(f'TP: {np.count_nonzero(TP)}')
    print(f'FN: {np.count_nonzero(FN)}')
    print(f'FP: {np.count_nonzero(FP)}')
    S= np.count_nonzero(TP)/cho_count
    print(f'S={S*100}%')
    print(f'Delay: {delay/cho_count}')

def plot_eval(df, y_label, y_pred, begin=0, end=0, title=''):
    if end==0:
        end=len(y_label)

    fig = plt.figure(figsize=(12, 8))
    fig.canvas.set_window_title(title)
    fig.suptitle(title)

    datetime = df['datetime'][begin:end]

    plt.subplot(2, 1, 1)
    plt.scatter(datetime, y_pred[begin:end]+1, label='predicted', s=6)
    plt.scatter(datetime, y_label[begin:end], label='CHO', s=6)
    plt.legend()
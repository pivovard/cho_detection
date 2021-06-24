"""
This script transpose measured data from the log file to the csv file,
where columns are signals and row is time.
No data modification is made.

@author Bc. David Pivovar
"""

import os

from numpy.core.records import array
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tabulate import tabulate

import alg.utils as utils

#transpose data from log file to csv file
def load_log(patientID=0, type='training', dir='', log_file='', verbose=True):
    if log_file == '':
        path = f'data/{patientID}-ws-{type}.log'
    else:
        path = dir+log_file

    print('Transfering data from log file:')
    print(path)
    
    file = pd.read_csv(path, sep=';')

    headers = file[' Signal'].unique()
    headers[0] = 'Device Time'
    for i in range(len(headers)):
        headers[i] = headers[i].strip()

    if verbose:
        print(headers)
        print(tabulate(file.head(20), headers='keys', tablefmt='psql'))

    data = pd.DataFrame(columns=headers)
    for index, row in file.iterrows():
        if verbose:
            utils.printProgressBar(index, len(file), length=50)

        dt = row[' Device Time']
        col = row[' Signal'].strip()
        val = row[' Info'].strip()

        if col is '':
            continue
        if len(data) == 0 or data['Device Time'].iat[-1] != dt:
            data = data.append({'Device Time': dt}, ignore_index=True)

        data[col].iat[-1] = val

    if dir=='':
        data.to_csv(f'data/{patientID}-transposed-{type}.csv', index=False, sep=';')
    else:
        data.to_csv(f'{dir}{log_file}.csv', index=False, sep=';')

    if verbose:
        print(tabulate(data.head(20), headers='keys', tablefmt='psql'))

#transpose multimple log files
def load_log_all(patientIDs=[], type='training', dir=''):
    utils.print_h('START')

    if len(patientIDs) > 0:
        for i, id in enumerate(patientIDs):
            load_log(patientID=id, type=type, verbose=True)
    else:
        for f in os.listdir(dir):
            load_log(dir=dir, file=f, verbose=True)

    utils.print_h('END')
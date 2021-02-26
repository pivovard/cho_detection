import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tabulate import tabulate 

import utils

def load_log(patientID):
    utils.printh('Transfering data from log file')

    path = f'data/{patientID}-ws-training.log'
    print(path)
    
    file = pd.read_csv(path, sep=';')
    
    headers = file[' Signal'].unique()
    headers[0] = 'Device Time'
    for i in range(len(headers)):
      headers[i] = headers[i].strip()
    
    print(headers)
    print(tabulate(file.head(20), headers = 'keys', tablefmt = 'psql')) 
    
    data = pd.DataFrame(columns=headers)
    
    utils.printProgressBar(0, len(file), length = 50)
    for index, row in file.iterrows():
      dt = row[' Device Time']
      col = row[' Signal'].strip()
      val = row[' Info'].strip()
    
      if col is '':
        continue
    
      if len(data) == 0 or data['Device Time'].iat[-1] != dt:
        data = data.append({'Device Time': dt}, ignore_index=True)
    
      data[col].iat[-1] = val
      utils.printProgressBar(index + 1, len(file), length = 50)
      
    data.to_csv(f'data/{patientID}-transposed.csv', index=False, sep=';')
    print(tabulate(data.head(20), headers = 'keys', tablefmt = 'psql')) 
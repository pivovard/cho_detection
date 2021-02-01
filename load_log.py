import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tabulate import tabulate 

# Print iterations progress
def printProgressBar (iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 100, fill = '=', printEnd = "\r"):
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
    # Print New Line on Complete
    #if iteration == total: 
    #    print()

def load_log(patientID):
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
    
    printProgressBar(0, len(file), length = 50)
    for index, row in file.iterrows():
      dt = row[' Device Time']
      col = row[' Signal'].strip()
      val = row[' Info'].strip()
    
      if col is '':
        continue
    
      if len(data) == 0 or data['Device Time'].iat[-1] != dt:
        data = data.append({'Device Time': dt}, ignore_index=True)
    
      data[col].iat[-1] = val
      printProgressBar(index + 1, len(file), length = 50)
      
    data.to_csv(f'data/{patientID}-transposed.csv', index=False, sep=';')
    print(tabulate(data.head(20), headers = 'keys', tablefmt = 'psql')) 
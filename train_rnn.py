"""
This script trains RNN keras model

@author Bc. David Pivovar
"""

import sys

import alg.load_log as lg
import alg.load_data as ld
import alg.cho_detection as cho

headers = ['ist', 'd1', 'minute_n']

## Tain patient individual model
def train(IDs, type):
    lg.load_log_all(IDs, 'training')

    for i, ID in enumerate(IDs):
        df = ld.load_data(ID, label='Interstitial glucose', fill_missing='',
                         smooth='savgol', derivation='difference', norm='',
                         verbose=True, graphs=True, analyze=False)
        cho.rnn(df, headers,'cho2', type, epochs=100, patientID=ID)
        
## Train model from all given training files
def train_all(IDs, type):
    lg.load_log_all(IDs, 'training')
    df = ld.load_data_all(IDs, from_file=False, fill_missing='', smooth='savgol', derivation='difference', norm='')
    cho.rnn(df, headers,'cho2', type, epochs=100, patientID=0)

def hint():
    print('python train_rnn.py <type> <option> [IDs]')
    print('Training files must be in /model/[ID]-ws-training.log.')
    print('Run with Python 3.7 or 3.8.')
    print('Type:')
    print('-gru')
    print('-lstm\n')
    print('Options:')
    print('-a\t train model from all given training files')
    print('-i\t train patient individual model')
    print('-h\t hint\n')

def main(argv):
    if len(sys.argv) < 4 or (sys.argv[1] != '-gru' and sys.argv[1] != '-lstm'):
        print('Invalid arguments!\n')
        hint()
        return
    
    type = sys.argv[1][1:]
    arg = sys.argv[2]
    if arg == '-a':
        train_all(sys.argv[3:], type)
    elif arg == '-i':
        train(sys.argv[3:], type)
    elif arg == '-h':
        hint()
    else:
        print('Invalid arguments!\n')
        hint()

if __name__ == "__main__":
   main(sys.argv[1:])
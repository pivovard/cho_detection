"""
This script trains RNN keras model

@author Bc. David Pivovar
"""

import sys, os, getopt
import pandas as pd

import alg.load_log as lg
import alg.load_data as ld
import alg.cho_detection as cho

headers = ['ist', 'd1', 'minute_n']

def hint():
    print('Run with Python 3.7 or 3.8.')
    print('python train_rnn.py <type> <options> [values]')
    print('Example of use:\n'\
        'Train GRU for all patients: python train_rnn.py -gru -f * -i data/ -o model/\n'\
        'Train LSTM for patient 575: python train_rnn.py -lstm -f 575-ws-training.log -i data/ -o model/\n')

    print('Type of the recurrent neurel network (mandatory):')
    print('-gru')
    print('-lstm\n')

    print('Options:')
    print('-f \t input file name (to train from all files in the folder use *)')
    print('-i\t input dir (default data/)')
    print('-o\t output dir (default model/)')
    print('-t\t use transposed data files (input folder must contain .log.csv files)')
    print('-m\t use modified data files (input folder must contain modified .csv files)')
    print('-h\t hint\n')

def main(argv):
    
    if not ('-gru' == argv[0] or '-lstm' == argv[0]):
        print('Invalid arguments!\n')
        hint()
        return

    type = argv[0][1:]
    file='*'
    input='data/'
    output='model/'
    transpose = True
    modify = True
    
    try:
        opts, args = getopt.getopt(argv[1:],"htmf:i:o:",[])
    except getopt.GetoptError:
        print('Invalid arguments!\n')
        hint()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            hint()
            sys.exit()
        if opt == '-f':
            file=arg
            if not os.path.exists(file):
                print('File doesn\'t exists!')
                exit()
        if opt == '-i':
            input=arg+'/'
        if opt == '-o':
            output=arg+'/'
        if opt == '-t':
            transpose = False
        if opt == '-m':
            modify = False

    print(f'Training {type}. File {input}{file}, output {output}.')

    if file == '*':
        print('Files ' + str(os.listdir(input)))
        # Transpose data
        if transpose:
            for f in os.listdir(input):
                if f[-4:] != '.log':
                    continue
                lg.load_log(dir=input, log_file=f, verbose=True)

        # Load modified data
        df=pd.DataFrame()
        for f in os.listdir(input):
            # Modify data
            if modify:
                if f[-8:] != '.log.csv':
                    continue
                d = ld.load_data(dir=input, file=f, label='Interstitial glucose', fill_missing='',
                                 smooth='savgol', derivation='difference', norm='',
                                 verbose=True, graphs=False, analyze=False)
                df=df.append(d)
            # Load saved data
            else:
                if f[-24:] != 'Interstitial glucose.csv':
                    continue
                d = ld.load_data_file(dir=input, file=f, label='Interstitial glucose')
                df=df.append(d)
        df=df.reset_index(drop=True)

        # Train RNN
        cho.rnn(df, headers,'cho2', type, epochs=80, path=output)
    else:
        # Transpose data
        if transpose:
            lg.load_log(dir=input, log_file=file)
        # Modify data
        df = ld.load_data(dir=input, file=(file+'.csv'), label='Interstitial glucose', fill_missing='',
                         smooth='savgol', derivation='difference', norm='',
                         verbose=True, graphs=False, analyze=False)
        # Train RNN
        cho.rnn(df, headers,'cho2', type, epochs=80, path=output)


if __name__ == "__main__":
   main(sys.argv[1:])
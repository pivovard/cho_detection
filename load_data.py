import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sweetviz

from scipy import interpolate

from tabulate import tabulate
from datetime import datetime
from datetime import timedelta

import utils

def to_datetime(df, column):
    device_time = df.pop(column)
    date_time = pd.to_datetime(device_time, format='%Y-%m-%d %H:%M:%S')
    df['datetime'] = date_time
    return df

#get timestamp and weekday
def process_time(df):
    time = df['datetime'].apply(lambda date : date.time())

    df['hour'] = time.apply(lambda t: t.hour)
    m = df['hour'].min()
    d = df['hour'].max() - m
    df['hour'] = df['hour'].apply(lambda t: (t-m)/d)
    
    df['quarter'] = time.apply(lambda t: t.hour*10 + (t.minute/40)*4)
    m = df['quarter'].min()
    d = df['quarter'].max() - m
    df['quarter'] = df['quarter'].apply(lambda t: (t-m)/d)

    df['timestamp'] = time.apply(lambda t: (t.hour * 60 + t.minute) * 60 + t.second)
    m = df['timestamp'].min()
    d = df['timestamp'].max() - m
    df['timestamp'] = df['timestamp'].apply(lambda t: (t-m)/d)

    df['weekday'] = df['datetime'].apply(lambda date: date.weekday())
    m = df['weekday'].min()
    d = df['weekday'].max() - m
    df['weekday'] = df['weekday'].apply(lambda t: (t-m)/d)

    return df

#drop null IST
def get_ist(df, fill_missing):
    print('\nProcessing IST...')
    df_ist = df[df['Interstitial glucose'].notna()].reset_index(drop=True)

    # add missing columns
    for i, column in enumerate(['Carbohydrate intake', 'Physical activity', 'requested insulin bolus', 'requested insulin basal rate', 'Steps']):
        if not column in df:
            df[column] = np.full(len(df), None)
            
    df = df[df['Interstitial glucose'].notna()
            | df['Carbohydrate intake'].notna()
            | df['Physical activity'].notna()
            | df['requested insulin bolus'].notna()
            | df['requested insulin basal rate'].notna()
            | df['Steps'].notna()]
    
    if fill_missing:
        print("\nFilling missing values...")
        delta = timedelta(minutes=10)
        mean = df_ist[utils.ist_l].mean()
        for i in range(len(df_ist)-1):
            d1= df_ist.loc[i, 'datetime']
            d2= df_ist.loc[i+1, 'datetime']
            diff = d2 - d1
            if diff > delta:
                count = int(diff / timedelta(minutes=5))
                dates = np.array([d1 + timedelta(minutes=i*5) for i in range(1, count)])
                df_tmp = pd.DataFrame([[np.nan for i in df.columns] for i in range(count-1)], columns=df.columns, dtype=float)
                df_tmp['datetime'] = pd.to_datetime(dates)
                # Set ist value
                ist = np.array([mean for i in range(count-1)])
                df_tmp[utils.ist_l] = ist
                print(f'Between {d1} and {d2} add {dates[0]} - {dates[-1]}')
                df = df.append(df_tmp, ignore_index=True)

    df = df.sort_values('datetime').reset_index(drop=True)

    #shift carb values matching nan ist
    #add processing for multiple columns as for cho
    delta = timedelta(minutes=20)
    for i, column in enumerate(['Carbohydrate intake', 'Physical activity', 'requested insulin bolus', 'requested insulin basal rate', 'Steps']):
        print(f'\nShifting {column}')
        for index, row in df.iterrows():
            if not pd.isnull(row[column]) and pd.isnull(row['Interstitial glucose']):
                date_time = row['datetime']
                value = row[column]
                print('Original datetime ' + str(row['datetime']))

                #assign to closer ist value
                if index > 0 and index < len(df)-1:
                    prev = date_time - df.loc[index-1, 'datetime']
                    next = df.loc[index+1, 'datetime'] - date_time
                    print('Prev = ' + str(prev) + ' Next = ' + str(next))

                    #if time delta greater than 20min
                    if prev > delta and next > delta:
                        print('No close ist value.')
                        continue

                    #assign to closer
                    if prev < next:
                        df.loc[index-1, column] = value
                    else:
                        df.loc[index+1, column] = value

                # Boundary values
                elif index == 0:
                    next = df.loc[index+1, 'datetime'] - date_time
                    if next < delta:
                        df.loc[index+1, column] = value
                else:
                    prev = date_time - df.loc[index-1, 'datetime']
                    if prev < delta:
                        df.loc[index-1, column] = value
                    
    # Clear null ist
    df = df[df['Interstitial glucose'].notna()]
    df = df.reset_index(drop=True)
    # df.pop('index') #delete old index column
    return df

def get_cho(df):
    print('\nProcessing CHO...')
    df['cho2'] = np.zeros(len(df))
    it = df.iterrows()
    for index, row in it:
        if row[utils.cho_l] > 0:
            for i in range(24): #2h
                df.loc[index+i, 'cho2'] = df.loc[index+i, 'cho2'] + row[utils.cho_l]
    #boolean values
    df['cho_b'] = df[utils.cho_l] > 0
    df['cho2_b'] = df['cho2'] > 0
    return df

#NaN -> 0, - -> 0
def clean_data(df):
    date_time = df.pop('datetime')

    #replace NaN values
    df_clean = df.fillna(0)
    #replace negative values
    #df_clean[df_clean < 0] = 0

    df_clean['datetime'] = date_time
    return df_clean

def normalize(df):
    print('\nNormalization')
    for i, column in enumerate(['Interstitial glucose', 'Carbohydrate intake', 'cho2', 'Physical activity', 'requested insulin bolus', 'requested insulin basal rate', 'Steps']):
        m = df[column].min()
        d = df[column].max() - m
        df[f'{column}'] = df[column].apply(lambda t: (t-m)/d)
    return df

def calc_derivations(df):
    print('\nCalculating derivations...')
    der = pd.DataFrame(columns=['der1', 'der2', 'der3'], dtype=float)
    delta = timedelta(minutes=5).total_seconds()
    for i in range(len(df)-1):
        utils.printProgressBar(i, len(df)-1)
        if (df.loc[i+1,'datetime']-df.loc[i,'datetime']) > timedelta(minutes=5) or df.loc[i,utils.ist_l] == 0 or df.loc[i+1,utils.ist_l] == 0:
            der.loc[i] = [0,0,0]
            continue
        der1=(df.loc[i+1,utils.ist_l]-df.loc[i,utils.ist_l])/delta
        der2=der1/delta
        der3=der2/delta
        der.loc[i] = [der1,der2,der3]

    # der = pd.concat([pd.DataFrame([0,0,0]), der], ignore_index=True)
    return pd.concat([df, der], axis=1)

def calc_gradient(df):
    print('\nCalculating gradient...')
    delta = timedelta(minutes=5).total_seconds()
    #delta = (1/24)/12
    df['grad1'] = np.gradient(df[utils.ist_l], delta, edge_order=2)
    df['grad2'] = np.gradient(df['grad1'], delta, edge_order=2)
    df['grad3'] = np.gradient(df['grad2'], delta, edge_order=2)
    return df

def plot_graph(df, begin=0, end=0, title='24h'):
    if end==0:
        end=len(df)
    datetime = df['datetime'][begin:end]

    fig = plt.figure(figsize=(12, 8))
    fig.canvas.set_window_title(title)
    fig.suptitle(title)
    #fig.autofmt_xdate()

    plt.subplot(3, 1, 1)
    plt.title('Interstitial glucose')
    plt.plot(datetime, df[utils.ist_l][begin:end], label=utils.ist_l)
    plt.scatter(datetime, df[utils.cho_l][begin:end], label=utils.cho_l, c='g', s=10)
    #plt.xticks(rotation=50)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.title('Carbohydrate intake, PA, Insulin')
    plt.scatter(datetime, df[utils.cho_l][begin:end], label=utils.cho_l, c='g', s=10)
    plt.scatter(datetime, df[utils.phy_l][begin:end], label=utils.phy_l, c='r', s=10)
    plt.scatter(datetime, df[utils.inb_l][begin:end], label=utils.inb_l, c='k', s=10)
    plt.scatter(datetime, df[utils.inr_l][begin:end], label=utils.inr_l, c='c', s=10)
    #plt.xticks(rotation=50)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.title('Carbohydrate intake')
    plt.scatter(datetime, df['cho2'][begin:end], label='cho2', s=6)
    plt.scatter(datetime, df[utils.cho_l][begin:end], label='cho', s=10)
    plt.legend()
    plt.ylabel('[g]')

def plot_derivations(df, der, grad, begin=0, end=0, title=''):
    if end==0:
        end=len(df)
    datetime = df['datetime'][begin:end]

    # Derivations
    if der:
        fig = plt.figure(figsize=(12, 8))
        fig.canvas.set_window_title('Derivations')
        fig.suptitle('Derivations')

        plt.subplot(4, 1, 1)
        plt.plot(datetime, df[utils.ist_l][begin:end], label='ist')
        plt.legend()
        plt.ylabel('Interstitial glucose [mmol/l]')

        plt.subplot(4, 1, 2)
        plt.plot(datetime, df['der1'][begin:end], label='der1 [mmol/l^2]')
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(datetime, df['der2'][begin:end], label='der2 [mmol/l^2]')
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(datetime, df['der3'][begin:end], label='der3 [mmol/l^4]')
        plt.legend()

    # Gradient
    if grad:
        fig = plt.figure(figsize=(12, 8))
        fig.canvas.set_window_title('Gradient '+ title)
        fig.suptitle('Gradient '+ title)

        plt.subplot(4, 1, 1)
        plt.plot(datetime, df[utils.ist_l][begin:end], label='ist')
        plt.legend()
        plt.ylabel('Interstitial glucose [mmol/l]')

        plt.subplot(4, 1, 2)
        plt.plot(datetime, df['grad1'][begin:end], label='grad1 [mmol/l^2]')
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(datetime, df['grad2'][begin:end], label='grad2 [mmol/l^2]')
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(datetime, df['grad3'][begin:end], label='grad3 [mmol/l^4]')
        plt.legend()

def load_data(patientID, from_file=False, fill_missing=False, norm=False, der=False, grad=True,
              verbose=True, graphs=False, analyze=False):
    if from_file:
        utils.print_h('Loading data from file.')
        df = pd.read_csv(f'data/{patientID}-modified.csv', sep=';')
        #set datetime type
        df = to_datetime(df, 'datetime')
    else:
        utils.print_h('Loading and modifying data from csv.')
        df = pd.read_csv(f'data/{patientID}-transposed.csv', sep=';')

        if verbose:
            print('Original data:')
            print(tabulate(df.head(20), headers = 'keys', tablefmt = 'psql'))
            print(df.describe().transpose(), end='\n\n')

        df = to_datetime(df, 'Device Time')
        df = get_ist(df, fill_missing)
        df = get_cho(df)
        df = process_time(df)

        # derivations
        if(der):
            df = calc_derivations(df)
        if(grad):
            df = calc_gradient(df)
        if(norm):
            df = normalize(df)

        df.to_csv(f'data/{patientID}-modified.csv', index=False, sep=';')

    if verbose:
        print('Training data:')
        print(tabulate(df.head(20), headers = 'keys', tablefmt = 'psql'))
        print(df.describe().transpose(), end='\n\n')

    if analyze:
        report = sweetviz.analyze(df)
        report.show_html()

    if graphs:
        plot_graph(df, title = 'Whole dataset') 
        #plot_graph(df, end=288, title = '24h dataset')
        plot_derivations(df, der, grad, title = 'Whole dataset')
        #plot_derivations(df, der, grad, end=288, title = '24h dataset')

    df = clean_data(df)
    return df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tabulate import tabulate
from datetime import timedelta

cho_l = 'Carbohydrate intake'
ist_l = 'Interstitial glucose'

#get timestamp and weekday
def process_time(df):
    device_time = df.pop('Device Time')
    date_time = pd.to_datetime(device_time, format='%Y-%m-%d %H:%M:%S')
    df['datetime'] = date_time

    time = date_time.apply(lambda date : date.time())
    df['hour'] = time.apply(lambda t: t.hour)
    df['timestamp'] = time.apply(lambda t: (t.hour * 60 + t.minute) * 60 + t.second)
    df['weekday'] = date_time.apply(lambda date: date.weekday())

    return df

#drop null IST
def get_ist(df):
    #df = df[df['Interstitial glucose'].notna()]
    df = df[df['Interstitial glucose'].notna() | df['Carbohydrate intake'].notna()]
    df = df.reset_index()

    #shift carb values matching nan ist
    #add processing for multiple columns as for cho
    print('Shifted carb intake')
    for index, row in df.iterrows():
        if not np.isnan(row['Carbohydrate intake']) and np.isnan(row['Interstitial glucose']):
            datetime = row['datetime']
            carb = row[cho_l]
            print('Original datetime ' + str(row['datetime']))

            #assign to closer ist value
            if index > 0 and index < len(df)-1:
                prev = datetime - df.loc[index-1, 'datetime']
                next = df.loc[index+1, 'datetime'] - datetime
                print('Prev = ' + str(prev) + ' Next = ' + str(next))

                #if value greater than 20min
                delta = timedelta(0, 0, 0, 0, 20, 0)
                if prev > delta or next > delta:
                    print('No close ist value.')
                    continue

                #assign to closer
                if prev < next:
                    df.loc[index-1, cho_l] = carb
                else:
                    df.loc[index+1, cho_l] = carb

            else:
                print('Boundary value.')

    df = df[df['Interstitial glucose'].notna()]
    df = df.reset_index()
    return df

#NaN -> 0, - -> 0
def clean_data(df):
    date_time = df.pop('datetime')

    #replace NaN values
    df_clean = df.fillna(0)
    #replace negative values
    df_clean[df_clean < 0] = 0

    df_clean['datetime'] = date_time
    return df_clean

def normalize():
    pass

def plot_graph(df):
    fig = plt.figure(figsize=(12, 8))
    # fig.autofmt_xdate()

    plt.subplot(2, 1, 1)
    plt.plot(df['datetime'], df[ist_l], label=ist_l)
    # plt.xticks(rotation=50)
    plt.legend()
    plt.title('Whole dataset')

    plt.subplot(2, 1, 2)
    plt.scatter(df['datetime'], df[cho_l], label=cho_l, c='g', s=10)
    # plt.xticks(rotation=50)
    plt.legend()
    

def plot_graph_part(df, begin=0, end=288):
    fig = plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(df['datetime'][begin:end], df[ist_l][begin:end], label=ist_l)
    plt.xticks(rotation=50)
    plt.legend()
    plt.title('24h')

    plt.subplot(2, 1, 2)
    plt.scatter(df['datetime'][begin:end], df[cho_l][begin:end], label=cho_l, c='g', s=10)
    plt.xticks(rotation=50)
    plt.legend()

def load_data(patientID, from_file=False, headers = [], normalize = False, verbose=False, graphs=False):
    if from_file:
        df = pd.read_csv(f'data/{patientID}-modified.csv', sep=';')
        #set datetime type
        device_time = df.pop('datetime')
        date_time = pd.to_datetime(device_time, format='%Y-%m-%d %H:%M:%S')
        df['datetime'] = date_time

    else:
        df = pd.read_csv(f'data/{patientID}-transposed.csv', sep=';')

        if verbose:
            print('Original data:')
            print(tabulate(df.head(20), headers = 'keys', tablefmt = 'psql'))
            print(df.describe().transpose(), end='\n\n')

        df = process_time(df)
        df = get_ist(df)

        df.to_csv(f'data/{patientID}-modified.csv', index=False, sep=';')

    if verbose:
        print('Training data:')
        print(tabulate(df.head(20), headers = 'keys', tablefmt = 'psql'))
        print(df.describe().transpose(), end='\n\n')

    if graphs:
        plot_graph(df)
        plot_graph_part(df)
        #plt.show()

    df = clean_data(df)
    return df
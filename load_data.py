import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sweetviz

from scipy import interpolate
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import savgol_filter

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
    df['hour_n'] = df['hour'].apply(lambda t: t/24)
    
    df['quarter'] = time.apply(lambda t: t.hour*10 + (t.minute/40)*4)
    df['quarter_n'] = df['quarter'].apply(lambda t: t/236)

    df['minute'] = time.apply(lambda t: t.hour * 60 + t.minute)
    df['minute_n'] = df['minute'].apply(lambda t: t/1440)

    df['timestamp'] = time.apply(lambda t: (t.hour * 60 + t.minute) * 60 + t.second)
    df['timestamp_n'] = df['timestamp'].apply(lambda t: t/86340)

    df['weekday'] = df['datetime'].apply(lambda date: date.weekday())
    df['weekday_n'] = df['weekday'].apply(lambda t: t/6)

    return df

#drop null IST
def get_ist(df, fill_missing):
    print('\nProcessing IST...')
    df_ist = df[df['Interstitial glucose'].notna()].reset_index(drop=True)

    # add missing columns
    headers = ['Carbohydrate intake', 'Physical activity', 'requested insulin bolus', 'requested insulin basal rate',
               'Acceleration', 'Steps', 'Heartbeat', 'Electrodermal activity', 'Skin temperature', 'Air temperature']
    for i, column in enumerate(headers):
        if not column in df:
            df[column] = np.full(len(df), None)
            
    df = df[df['Interstitial glucose'].notna()
            | df['Carbohydrate intake'].notna()
            | df['Physical activity'].notna()
            | df['requested insulin bolus'].notna()
            | df['requested insulin basal rate'].notna()
            | df['Steps'].notna()
            | df['Heartbeat'].notna()
            | df['Electrodermal activity'].notna()
            | df['Skin temperature'].notna()
            | df['Air temperature'].notna()
            | df['Acceleration'].notna()]
    
    if fill_missing != '':
        print(f'Filling missing values: {type}')
        delta = timedelta(minutes=10)
        mean = df_ist[utils.ist_l].mean()
        spline = None
        if fill_missing=='akima':
            spline = Akima1DInterpolator(df_ist['datetime'],df_ist[utils.ist_l])
        if fill_missing=='min':
            delta = timedelta(minutes=2)
            spline = Akima1DInterpolator(df_ist['datetime'],df_ist[utils.ist_l])

        for i in range(len(df_ist)-1):
            d1= df_ist.loc[i, 'datetime']
            d2= df_ist.loc[i+1, 'datetime']
            diff = d2 - d1
            if diff > delta:
                count=0
                dates=[]
                if fill_missing=='min':
                    count = int(diff / timedelta(minutes=1))
                    dates = np.array([d1 + timedelta(minutes=i*1) for i in range(1, count)])
                else:
                    count = int(diff / timedelta(minutes=5))
                    dates = np.array([d1 + timedelta(minutes=i*5) for i in range(1, count)])
                df_tmp = pd.DataFrame([[np.nan for i in df.columns] for i in range(count-1)], columns=df.columns, dtype=float)
                df_tmp['datetime'] = pd.to_datetime(dates)
                # Set ist value
                if fill_missing == 'mean':
                    ist = np.array([mean for i in range(count-1)])
                elif fill_missing == 'akima' or fill_missing=='min':
                    ist = spline(df_tmp['datetime'])
                df_tmp[utils.ist_l] = ist
                # print(f'Between {d1} and {d2} add {dates[0]} - {dates[-1]}')
                df = df.append(df_tmp, ignore_index=True)

    df = df.sort_values('datetime').reset_index(drop=True)

    #shift carb values matching nan ist
    #add processing for multiple columns as for cho
    delta = timedelta(minutes=20)
    for i, column in enumerate(headers):
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
    df['cho'] = df['Carbohydrate intake']
    df['cho2'] = np.zeros(len(df))
    df['cho_cat'] = np.zeros(len(df))
    it = df.iterrows()
    for index, row in it:
        # copy 2h
        if row[utils.cho_l] > 0:
            for i in range(24): #2h
                if (index+i)>=len(df):
                    break
                df.loc[index+i, 'cho2'] = df.loc[index+i, 'cho2'] + row[utils.cho_l]
        # categories
        if row[utils.cho_l] > 0:
            df.loc[index+i, 'cho_cat'] = 1
        if row[utils.cho_l] > 25:
            df.loc[index+i, 'cho_cat'] = 2
        elif row[utils.cho_l] > 50:
            df.loc[index+i, 'cho_cat'] = 3
        elif row[utils.cho_l] > 75:
            df.loc[index+i, 'cho_cat'] = 4
    #boolean values
    df['cho_b'] = df[utils.cho_l] > 0
    df['cho2_b'] = df['cho2'] > 0
    return df

def get_pa(df):
    print('\nProcessing PA...')
    df['pa'] = np.zeros(len(df))
    it = df.iterrows()
    for index, row in it:
        if pd.notna(row[utils.phy_l]) and row[utils.phy_l] > 0:
            for i in range(12): #2h
                if (index+i)>=len(df) or df.loc[index+i, utils.cho_l] == 0:
                    break
                df.loc[index+i, 'pa'] = row[utils.cho_l]
    return df

#NaN -> 0, - -> 0
def replace_nan(df):
    date_time = df.pop('datetime')

    #replace NaN values
    df_clean = df.fillna(0)
    #replace negative values
    #df_clean[df_clean < 0] = 0

    df_clean['datetime'] = date_time
    return df_clean

def normalize(df, type):
    print(f'\nNormalization: {type}')
    if type == 'std':
        # for i, column in enumerate(['Interstitial glucose', 'ist', 'Carbohydrate intake', 'cho2', 'Physical activity', 'requested insulin bolus', 'requested insulin basal rate', 'Steps']):
        for i, column in enumerate(['Interstitial glucose', 'ist']):
            mean = df[column].mean()
            std = df[column].std()
            df[f'{column}'] = df[column].apply(lambda t: (t-mean)/std)
    elif type == 'minmax':
        for i, column in enumerate(['Interstitial glucose', 'ist', 'Carbohydrate intake', 'cho2', 'Physical activity', 'requested insulin bolus', 'requested insulin basal rate', 'Steps']):
            m = df[column].min()
            d = df[column].max() - m
            df[f'{column}'] = df[column].apply(lambda t: (t-m)/d)
    return df

def calc_derivations(df, type):
    print(f'\nCalculating derivations: {type}')
    if type == 'akima':
        akima = Akima1DInterpolator(df['datetime'], df['ist'])
        der1 = akima.derivative(1)
        df['d1'] = der1(df['datetime'])
        der2 = akima.derivative(2)
        df['d2'] = der2(df['datetime'])
        der3 = akima.derivative(3)
        df['d3'] = der3(df['datetime'])
    elif type == 'splrep':
        tck = interpolate.splrep(range(len(df)), df['ist'])
        df['d1'] = interpolate.splev(range(len(df)), tck, der=1)
        df['d2'] = interpolate.splev(range(len(df)), tck, der=2)
        df['d3'] = interpolate.splev(range(len(df)), tck, der=3)
    elif type == 'gradient':
        delta = 5
        df['d1'] = np.gradient(df['ist'], delta, edge_order=2)
        df['d2'] = np.gradient(df['d1'], delta, edge_order=2)
        df['d3'] = np.gradient(df['d2'], delta, edge_order=2)
    elif type == 'difference':
        der = pd.DataFrame(columns=['d1', 'd2', 'd3'], dtype=float)
        for i in range(len(df)-2):
            utils.printProgressBar(i, len(df)-2)
            if (df.loc[i+1,'datetime']-df.loc[i,'datetime']) > timedelta(minutes=5) or df.loc[i,'ist'] == 0 or df.loc[i+1,'ist'] == 0:
                der.loc[i] = [0,0,0]
                continue
            diff=df.loc[i+1,'ist']-df.loc[i,'ist']
            delta=df.loc[i+1,'datetime']-df.loc[i,'datetime']
            delta=delta.seconds/60
            d1=diff/delta
            d2=(df.loc[i+2,'ist']-df.loc[i+1,'ist']-diff)/(delta*delta)
            d3=0
            der.loc[i] = [d1,d2,d3]
        df = pd.concat([df, der], axis=1)
    return df

def plot_graph(df, begin=0, end=0, title=''):
    if end==0:
        end=len(df)
    datetime = df['datetime'][begin:end]

    fig = plt.figure(figsize=(12, 9))
    fig.canvas.set_window_title(title)
    fig.suptitle(title)
    #fig.autofmt_xdate()

    plt.subplot(3, 1, 1)
    plt.title('Interstitial glucose')
    plt.plot(datetime, df[utils.ist_l][begin:end], label=utils.ist_l)
    plt.plot(datetime, df['ist'][begin:end], label='IST smoothed')
    plt.scatter(datetime, 0.2*df[utils.cho_l][begin:end], label=f'{utils.cho_l} [g]', c='g', s=10)
    plt.scatter(datetime, df['pa'][begin:end], label=utils.ist_l)
    #plt.xticks(rotation=50)
    plt.legend()
    plt.ylabel('[mmol/l]')
    
    plt.subplot(3, 1, 2)
    plt.title('Carbohydrate intake, PA, Insulin bazal and bolus')
    plt.plot(datetime, np.zeros(len(datetime)), c='w') # dummy
    plt.scatter(datetime, df[utils.cho_l][begin:end], label=f'{utils.cho_l} [g]', c='g', s=10)
    plt.scatter(datetime, df[utils.phy_l][begin:end], label=utils.phy_l, c='r', s=10)
    plt.scatter(datetime, df[utils.inb_l][begin:end], label=f'{utils.inb_l} [UI]', c='k', s=10)
    plt.scatter(datetime, df[utils.inr_l][begin:end], label=f'{utils.inr_l} [UI/H]', c='c', s=10)
    #plt.xticks(rotation=50)
    plt.legend()

    # plt.subplot(3, 1, 3)
    # plt.title('Carbohydrate intake')
    # plt.scatter(datetime, df['cho2'][begin:end], label='cho2', s=6)
    # plt.scatter(datetime, df[utils.cho_l][begin:end], label='cho', s=10)
    # plt.legend()
    # plt.ylabel('[g]')

    plt.subplot(3, 1, 3)
    plt.title('Heartbeat, Electrodermal activity, Skin temperature, Air temperature')
    plt.plot(datetime, np.zeros(len(datetime)), c='w') # dummy
    plt.plot(datetime, df['Heartbeat'][begin:end], label='Heartbeat [bps]')
    plt.plot(datetime, df['Electrodermal activity'][begin:end]*10, label='Electrodermal activity [uS]')
    plt.plot(datetime, df['Skin temperature'][begin:end], label='Skin temperature [°C]')
    plt.plot(datetime, df['Air temperature'][begin:end], label='Air temperature [°C]')
    plt.plot(datetime, df['Steps'][begin:end], label='Steps')
    plt.legend()

    plt.tight_layout()
    plt.xlabel('time')

def plot_derivations(df, begin=0, end=0, title=''):
    if end==0:
        end=len(df)
    datetime = df['datetime'][begin:end]

    # Derivations
    fig = plt.figure(figsize=(12, 8))
    fig.canvas.set_window_title(f'Derivations {title}')
    fig.suptitle(f'Derivations {title}')
    plt.subplot(4, 1, 1)
    plt.plot(datetime, df[utils.ist_l][begin:end], label='ist')
    plt.legend()
    plt.ylabel('Interstitial glucose [mmol/l]')
    plt.subplot(4, 1, 2)
    plt.plot(datetime, df['d1'][begin:end], label='d1 [mmol/l/t^2]')
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(datetime, df['d2'][begin:end], label='d2 [mmol/l/t^3]')
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(datetime, df['d3'][begin:end], label='d3 [mmol/l/t^4]')
    plt.legend()

def load_data(patientID, from_file=False, fill_missing='', smooth='', derivation='', norm='',
              verbose=True, graphs=False, analyze=False):
    if from_file:
        utils.print_h(f'Loading data from file {patientID}')
        df = pd.read_csv(f'data/{patientID}-modified.csv', sep=';')
        #set datetime type
        df = to_datetime(df, 'datetime')
    else:
        utils.print_h(f'Loading and modifying data from csv {patientID}')
        df = pd.read_csv(f'data/{patientID}-transposed.csv', sep=';')

        if verbose:
            print('Original data:')
            print(tabulate(df.head(20), headers = 'keys', tablefmt = 'psql'))
            print(df.describe().transpose(), end='\n\n')

        df = to_datetime(df, 'Device Time')
        df = get_ist(df, fill_missing)
        df = get_cho(df)
        df = get_pa(df)
        df = process_time(df)

        if smooth == 'savgol':
            df['ist'] = savgol_filter(df['Interstitial glucose'], 21, 3) # window size 51, polynomial order 3
        else:
            df['ist'] = df['Interstitial glucose']
        if derivation != '':
            df = calc_derivations(df, derivation)
        if norm != '':
            df = normalize(df, norm)

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
        plot_graph(df, end=288, title = '24h dataset')
        plot_derivations(df, title = 'Whole dataset')
        plot_derivations(df, end=288, title = '24h dataset')
        # example patientID 575
        plot_graph(df, begin=5*288-30, end=7*288-30, title = '48h dataset')

    # df = replace_nan(df)
    return df

def load_data_all(patientIDs, from_file, fill_missing='', smooth='', derivation='', norm=''):
    utils.print_h('START')

    dfs=pd.DataFrame()
    for i, id in enumerate(patientIDs):
        d=load_data(patientID=id, from_file=from_file,
                    fill_missing=fill_missing, smooth=smooth, derivation=derivation, norm=norm)
        dfs=dfs.append(d)
    dfs=dfs.reset_index(drop=True)
    print(tabulate(dfs.head(20), headers = 'keys', tablefmt = 'psql'))

    utils.print_h('END')

    return dfs
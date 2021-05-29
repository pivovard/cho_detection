from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import statistics

import sweetviz

import subprocess
from datetime import timedelta
import utils

def plot_pa(data, headers):
    datetime = data['datetime']

    fig = plt.figure(figsize=(12, 9))
    fig.canvas.set_window_title('Whole dataset')
    plt.subplot(5, 1, 1)
    plt.title('Value')
    for i, val in enumerate(headers):
        plt.plot(datetime, data[val], label=val)
    plt.scatter(datetime, data['pa'], label='PA', s=20, c='r')
    plt.legend()
    plt.subplot(5, 1, 2)
    plt.title('Mean')
    for i, val in enumerate(headers):
        plt.plot(datetime, data[f'{val} mean'], label=val)
    plt.scatter(datetime, data['pa'], label='PA', s=20, c='r')
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.title('Std')
    for i, val in enumerate(headers):
        plt.plot(datetime, data[f'{val} std'], label=val)
    plt.scatter(datetime, data['pa'], label='PA', s=20, c='r')
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.title('Median')
    for i, val in enumerate(headers):
        plt.plot(datetime, data[f'{val} median'], label=val)
    plt.scatter(datetime, data['pa'], label='PA', s=20, c='r')
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.title('Kvartil diff')
    for i, val in enumerate(headers):
        plt.plot(datetime, data[f'{val} kvartil'], label=val)
    plt.scatter(datetime, data['pa'], label='PA', s=20, c='r')
    plt.legend()
    plt.tight_layout()


    fig = plt.figure(figsize=(12, 9))
    fig.canvas.set_window_title('48h dataset')
    plt.subplot(5, 1, 1)
    plt.title('Value')
    for i, val in enumerate(headers):
        plt.plot(datetime[900:1600], data[val][900:1600], label=val)
    plt.scatter(datetime[900:1600], data['pa'][900:1600], label='PA', s=20, c='r')
    plt.legend()
    plt.subplot(5, 1, 2)
    plt.title('Mean')
    for i, val in enumerate(headers):
        plt.plot(datetime[900:1600], data[f'{val} mean'][900:1600], label=val)
    plt.scatter(datetime[900:1600], data['pa'][900:1600], label='PA', s=20, c='r')
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.title('Std')
    for i, val in enumerate(headers):
        plt.plot(datetime[900:1600], data[f'{val} std'][900:1600], label=val)
    plt.scatter(datetime[900:1600], data['pa'][900:1600], label='PA', s=20, c='r')
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.title('Median')
    for i, val in enumerate(headers):
        plt.plot(datetime[900:1600], data[f'{val} median'][900:1600], label=val)
    plt.scatter(datetime[900:1600], data['pa'][900:1600], label='PA', s=20, c='r')
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.title('Kvartil diff')
    for i, val in enumerate(headers):
        plt.plot(datetime[900:1600], data[f'{val} kvartil'][900:1600], label=val)
    plt.scatter(datetime[900:1600], data['pa'][900:1600], label='PA', s=20, c='r')
    plt.legend()
    plt.tight_layout()

def get_pa(patientID, df, headers):
    print('\nCalculating PA statistics')
    data = df[headers]
    w=6
    dlen=len(data)-w
    hlen=len(headers)

    mean = np.zeros((dlen,hlen))
    median = np.zeros((dlen,hlen))
    std = np.zeros((dlen,hlen))
    kvartil = np.zeros((dlen,hlen))
    
    for i in range(dlen):
        utils.printProgressBar(i, dlen)
        mean[i] = data[i:i+w].mean()
        std[i] = data[i:i+w].std()
        median[i] = data[i:i+w].median()
        q25 = data[i:i+w].quantile(0.25)
        q75 = data[i:i+w].quantile(0.75)
        kvartil[i] = q75 - q25

    data = data[w:]
    data['datetime'] = df['datetime'][w:]
    data['pa'] = df['pa'][w:]
    data['pa2'] = df['pa2'][w:]
    for i, val in enumerate(headers):
        data[f'{val} mean']=mean[:,i]
        data[f'{val} std']=std[:,i]
        data[f'{val} median']=median[:,i]
        data[f'{val} kvartil']=kvartil[:,i]

    data.to_csv(f'data/{patientID}-pa.csv', index=False, sep=';')

    plot_pa(data, headers)
    # report = sweetviz.analyze(data)
    # report.show_html()

    return data

def predict_pa(patientID, headers):
    df = pd.read_csv(f'data/{patientID}-pa.csv', sep=';')

    print('Mean')
    print(df[df['pa2']>0].mean())
    print('\nMedian')
    print(df[df['pa2']>0].median())
    print('\nQuantil 25%')
    print(df[df['pa2']>0].quantile(0.25))

    #get thresholds
    th = df[df['pa2']>0].mean()
    # th = df[df['pa']>0].quantile(0.85)

    res = np.zeros(len(df))
    for i, val in enumerate(headers):
        # res += np.array(df[f'{val}'] >= th[f'{val}'], dtype=int)
        res += np.array(df[f'{val} mean'] >= th[f'{val} mean'], dtype=int)
        # res += np.array(df[f'{val} std'] >= th[f'{val} std'], dtype=int)
        res += np.array(df[f'{val} median'] >= th[f'{val} median'], dtype=int)
        # res += np.array(df[f'{val} kvartil'] >= th[f'{val} kvartil'], dtype=int)

    utils.evaluate(df['pa'], res, 3)













def create_window(df, y_label, width):
    #shape (number of windows, window width, columns count)
    X = np.empty((len(df)-width, width, len(df.columns)))
    y = y_label[width:]
    for i in range(len(df)-width):
        X[i] = df[i:i+width]
    return X, y

def dense(patientID, headers, label):
    df = pd.read_csv(f'data/{patientID}-pa.csv', sep=';')
    df_train = df[:int(len(df)*0.8)].reset_index(drop=True)
    df_test = df[int(len(df)*0.8):].reset_index(drop=True)

    X = df_train[headers]
    y = df_train[label]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=len(headers), activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    model.fit(X, y, epochs=20, batch_size= 64,  shuffle=False)

    model.summary()

    lstm_test(df, headers, label, 15, model=model)

def conv(df, headers):
    pass

def lstm(df, headers, label, type, width=utils.WINDOW_WIDTH_1H*2, epochs=100, patientID=''):
    df_train = df[:int(len(df)*0.8)].reset_index(drop=True)
    df_test = df[int(len(df)*0.8):].reset_index(drop=True)

    X, y = create_window(df_train[headers], df_train[label], width)
    X_val, y_val = create_window(df_test[headers], df_test[label], width)

    model = tf.keras.Sequential()
    if type=='LSTM':
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=128,
                input_shape=[X.shape[1], X.shape[2]]
            )))
    elif type=='GRU':
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                units=128,
                input_shape=[X.shape[1], X.shape[2]]
            )))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    model.fit(X, y, epochs=epochs, batch_size= 64,  shuffle=False, validation_data=(X_val, y_val))

    model.summary()

    model.save(f'model/{patientID}_keras_model.h5')
    # try:
    #     subprocess.call(['python', 'convert_model.py', f'model/{patientID}_keras_model.h5', f'model/{patientID}_fdeep_model.json'])
    # except:
    #     print('Exception in convert script.')

    lstm_test(df, headers, label, 15, model=model)

    return model

def lstm_test(df, headers, label, th, model = None, path=None):
    df = df.reset_index(drop=True)
    if model is None:
        model = tf.keras.models.load_model(path)
    
    X, y = create_window(df[headers], df[label], utils.WINDOW_WIDTH_1H*2)
    y_pred = model(X)

    utils.evaluate(y, y_pred, th, 'LSTM')
    utils.plot_eval(df, y, y_pred, title='LSTM')
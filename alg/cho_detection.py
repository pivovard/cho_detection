"""
Train and evaluate algorithms for carbohydrate detection.

- RNN (keras model)
- LDA/GDA
- Edge detection

@author Bc. David Pivovar
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import colors

from scipy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import subprocess
from datetime import timedelta
import alg.utils as utils

## Split data into the sliding window
def create_window(df, y_label, width):
    #shape (number of windows, window width, columns count)
    X = np.empty((len(df)-width, width, len(df.columns)))
    y = y_label[width:]
    for i in range(len(df)-width):
        X[i] = df[i:i+width]
    return X, y

## Split data into the sliding window (shifted reference values)
def create_window_shifted(df, y_label, width, shift):
    #shape (number of windows, window width, columns count)
    X = np.empty((len(df)-width, width, len(df.columns)))
    y = y_label[int(width-shift):-int(shift)]
    for i in range(len(df)-width):
        X[i] = df[i:i+width]
    return X, y

## Train RNN
def rnn(df, headers, label, type, width=utils.WINDOW_WIDTH_1H*2, epochs=100, patientID=''):
    df=df.fillna(0)
    df_train = df[:int(len(df)*0.8)].reset_index(drop=True)
    df_test = df[int(len(df)*0.8):].reset_index(drop=True)

    X, y = create_window(df[headers], df[label], width)
    X_val, y_val = create_window(df_test[headers], df_test[label], width)

    model = tf.keras.Sequential()
    if type=='lstm':
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=128,
                input_shape=[X.shape[1], X.shape[2]]
            )))
    elif type=='gru':
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                units=128,
                input_shape=[X.shape[1], X.shape[2]]
            )))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    model.fit(X, y, epochs=epochs, batch_size= 64,  shuffle=False)

    model.summary()

    model.save(f'model/{patientID}_keras_model.h5')
    try:
        subprocess.call(['python', 'convert_model.py', f'model/{patientID}_keras_model.h5', f'model/{patientID}_fdeep_model.json'])
    except:
        print('Exception in convert script.')

    return model

## Evaluate RNN
def rnn_test(df, headers, label, th, model = None, path=None):
    df = df.reset_index(drop=True)
    if model is None:
        model = tf.keras.models.load_model(path)
    
    X, y = create_window(df[headers], df[label], utils.WINDOW_WIDTH_1H*2)
    y_pred = model(X)

    utils.evaluate(y, y_pred, th, 'LSTM')
    utils.plot_eval(df, y, y_pred, title='LSTM')

## Stack data into the sliding window
def window_stack(a, stepsize=1, width=12):
    n = a.shape[0]
    return np.hstack( a[i:1+n+i-width:stepsize] for i in range(0,width) )

## Train LDA/QDA with sliding window
def lda_window(df, header, width, title):
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)

    df_train = df[:int(len(df)*0.8)].reset_index(drop=True).fillna(0)
    df_test = df[int(len(df)*0.8):].reset_index(drop=True).fillna(0)

    # df_train['product'] = np.ones(len(df_train))
    # df_test['product'] = np.ones(len(df_test))
    # for i, col in enumerate(headers):
    #     df_train['product'] = df_train['product'] * df_train[col]    
    #     df_test['product'] = df_test['product'] * df_test[col]

    X = window_stack(df_train[[header]], width=width)    
    # X = window_stack(df_train[['product']], width=width)
    y = df_train['cho2_b'][width-1:]
    print("Input shape" + str(X.shape))

    lda.fit(X, y)
    qda.fit(X, y)

    X = window_stack(df_test[[header]], width=width)
    # X = window_stack(df_test[['product']], width=width)
    y = df_test['cho_b'][width-1:]
    y_pred=lda.predict(X)
    utils.evaluate(y, y_pred, 0, f'LDA window of {header}')
    utils.plot_eval(df_test, y, y_pred, title=f'LDA window of {header}')
    y_pred=qda.predict(X)
    utils.evaluate(y, y_pred, 0, f'QDA window of {header}')
    utils.plot_eval(df_test, y, y_pred, title=f'QDA window of {header}')

    return lda, qda

## Train LDA/QDA with multiple input data
def lda(df, headers, title):
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    
    df_train = df[:int(len(df)*0.8)].reset_index(drop=True).fillna(0)
    df_test = df[int(len(df)*0.8):].reset_index(drop=True).fillna(0)

    lda.fit(df_train[headers], df_train['cho2_b'])
    qda.fit(df_train[headers], df_train['cho2_b'])

    y_pred=lda.predict(df_test[headers])
    y=df_test['cho_b']
    utils.evaluate(y, y_pred, 0, 'LDA '+title)
    utils.plot_eval(df_test, y, y_pred, title='LDA '+title)
    y_pred=qda.predict(df_test[headers])
    utils.evaluate(y, y_pred, 0, 'QDA '+title)
    utils.plot_eval(df_test, y, y_pred, title='QDA '+title)

    # plot areas
    if len(headers) == 2:
        cho_true = df_test[df_test['cho2_b'] == True]
        cho_false = df_test[df_test['cho_b'] == False]

        fig = plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.suptitle('LDA')
        plt.scatter(cho_false[headers[0]], cho_false[headers[1]], label='CHO false', s=8, marker='o')
        plt.scatter(cho_true[headers[0]], cho_true[headers[1]], label='CHO true', s=15, marker='o')

        nx, ny = 200, 100
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                             np.linspace(y_min, y_max, ny))
        Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()+1/1000000000000])
        Z = Z[:, 1].reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap='RdBu',
                       norm=colors.Normalize(0., 1.), zorder=0)
        plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.suptitle('QDA')
        plt.scatter(cho_false[headers[0]], cho_false[headers[1]], label='CHO false', s=3, marker='o')
        plt.scatter(cho_true[headers[0]], cho_true[headers[1]], label='CHO true', s=5, marker='x')
        nx, ny = 200, 100
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                             np.linspace(y_min, y_max, ny))
        Z = qda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, 1].reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap='RdBu',
                       norm=colors.Normalize(0., 1.), zorder=0)
        plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')
        plt.legend()

    return lda, qda

## Edge detection
def threshold(df=None, th = [0.0125, 0.018], weight = [2.25,3]):
    df = df.reset_index(drop=True)
    N=len(df)
    
    datetime = df['datetime']

    flagsD1 = np.zeros((len(th),N))
    flagsD2 = np.zeros((len(th),N))
    flagsM1 = np.zeros((len(th),N))
    for i, val in enumerate(th):
        flagsD1[i] = (df['d1'] >= th[i]) * weight[i]
        flagsM1[i] = (df['d1'] <= th[i]*-1) * weight[i] * -1

    activation = np.max(flagsD1, axis=0)
    activation2 = np.max(flagsD1, axis=0)
    activation_m = np.min(flagsM1, axis=0)
    
    for i in range(24, N-1):
        if activation[i] > 2:
            for j in range(1, 25):
                if activation[i-j] >= 2+0.2*j:
                    activation[i] = activation[i] + 0.1*j
                    activation2[i] = activation2[i] + 0.1*j
        
        elif activation_m[i] < -2:
            activation2[i] = np.max(activation2[i-6:i])
            for j in range(1, 25):
                if activation_m[i-j] <= -1*(2+0.2*j):
                    activation_m[i] = activation_m[i] - 0.1*j
                    activation2[i] = activation2[i] - 0.1*j


    detected=np.full(N, None)
    detected2=np.full(N, None)
    detected_m=np.full(N, None)
    for i in range(12, len(df)):
        if activation[i] >= 2:
            detected[i] = activation[i]
        if activation2[i] >= 2:
            detected2[i] = activation2[i]
        if activation_m[i] <= -3:
            detected_m[i] = activation_m[i]

    fig = plt.figure(figsize=(12, 8))
    fig.canvas.set_window_title("Treshold")

    plt.subplot(3, 1, 1)
    plt.plot(datetime, df['ist'], label='Intersticiální glukóza')
    plt.scatter(datetime, df['cho']*0.2, label='Karbohydráty *0.2', s=15, c='g')
    plt.scatter(datetime, detected2, label='Klesající hrana', s=10, c='b', marker='*')
    plt.scatter(datetime, detected, label='Rostoucí hrana', s=10, c='r', marker='*')
    # plt.scatter(datetime, detected_m, label='decrease', s=10, c='b', marker='*')
    plt.xlabel('čas [mm-dd hh]')
    plt.ylabel('mmol/l')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(datetime, np.full(len(df), 0.018), label='treshold 0.018')
    plt.plot(datetime, np.full(len(df), 0.0125), label='treshold 0.0125')
    plt.plot(datetime, df['d1'], label='d1')
    # plt.plot(datetime, df['d2'], label='d2')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(datetime, activation, label='activation')
    plt.legend()
    plt.tight_layout()

    return activation
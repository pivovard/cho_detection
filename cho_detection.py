import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

from scipy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder

from datetime import timedelta
import utils

# model = nn_cho.compile(df, headers, utils.cho_l, utils.WINDOW_WIDTH_1H*3)

def plot_eval(y_label, y_pred, TP, FN, FP, begin=0, end=0, title=''):
    if end==0:
        end=len(y_label)

    fig = plt.figure(figsize=(12, 8))
    fig.canvas.set_window_title(title)
    fig.suptitle(title)

    plt.subplot(4, 1, 1)
    plt.title('Predicted CHO')
    plt.scatter(range(end-begin), y_pred[begin:end], label='predicted', s=6)
    plt.scatter(range(end-begin), y_label[begin:end], label='label', s=6)
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.title('TP')
    plt.scatter(range(end-begin), TP[begin:end], s=6)

    plt.subplot(4, 1, 3)
    plt.title('FN')
    plt.scatter(range(end-begin), FN[begin:end], s=6)

    plt.subplot(4, 1, 4)
    plt.title('FP')
    plt.scatter(range(end-begin), FP[begin:end], s=6)


def evaluate(y_label, y_pred, treshold=0, method='', graph=True):
    n = len(y_label)
    TP = np.zeros(n)
    FN = np.zeros(n)
    FP = np.zeros(n)
    delay = 0
    w = 24 #2h
    y_elements = []

    for i, y in enumerate(y_label):
        if y > 0:
            if n - i >= w:
                y_elements = y_pred[i:i+w]
            else:
                y_elements = y_pred[i:]
            if np.any(y_elements > treshold):
                TP[i] = True
            else:
                FN[i] = True
        elif y_pred[i] > treshold:
            if i >= w:
                y_elements = y_label[i-w:i]
            else:
                y_elements = y_label[:i]
            if not np.any(y_elements > 0) and np.all(FP[i-12:i]==False):
                FP[i] = True
    
    print(method)
    print(f'TP: {np.count_nonzero(TP)}')
    print(f'FN: {np.count_nonzero(FN)}')
    print(f'FP: {np.count_nonzero(FP)}')
    S= np.count_nonzero(TP)/(np.count_nonzero(TP)+np.count_nonzero(FN))
    print(f'S={S*100}%')

    if graph:
        plot_eval(y_label, y_pred, TP, FN, FP, title=f'{method}: All predicted values')
        # plot_eval(y_label, y_pred, TP, FN, FP, end=utils.WINDOW_WIDTH_24H*2, title=f'{method}: 48h predicted')

def create_window(df, y_label, width):
    #shape (number of windows, window width, columns count)
    X = np.empty((len(df)-width, width, len(df.columns)))
    y = y_label[width:]
    for i in range(len(df)-width):
        X[i] = df[i:i+width]
    return X, y

def create_window2(df, y_label, width, shift):
    #shape (number of windows, window width, columns count)
    X = np.empty((len(df)-width, width, len(df.columns)))
    y = y_label[int(width-shift):-int(shift)]
    for i in range(len(df)-width):
        X[i] = df[i:i+width]
    return X, y

def window_stack(a, stepsize=1, width=12):
    n = a.shape[0]
    return np.hstack( a[i:1+n+i-width:stepsize] for i in range(0,width) )

def dense(df):
    df_train = df[:int(len(df)*0.8)]
    df_test = df[int(len(df)*0.8):]
    headers = [utils.ist_l, 'hour', 'weekday', 'der1', 'grad1', 'grad2', 'grad3']

    X, y = create_window2(df_train[headers], df_train['cho2'], utils.WINDOW_WIDTH_1H*2, 6)

    model = tf.keras.Sequential([
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])

    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    model.fit(X, y, epochs=100, batch_size= 32,  shuffle=False)

    X, y = create_window(df_test[headers], df_test['Carbohydrate intake'], utils.WINDOW_WIDTH_1H*2)
    y_pred = model(X)

    evaluate(y, y_pred, 15, 'Dense', graph=True)


def conv(df):
    df_train = df[:int(len(df)*0.8)]
    df_test = df[int(len(df)*0.8):]
    headers = [utils.ist_l, 'quarter', 'weekday', 'der1', 'grad1', 'grad2', 'grad3']

    X, y = create_window2(df_train[headers], df_train['cho2'], utils.WINDOW_WIDTH_1H*2, 6)

    model = tf.keras.models.Sequential([
		tf.keras.layers.Conv1D(filters=3,
                           kernel_size=(utils.WINDOW_WIDTH_1H*2,),
                           activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1)
	])

    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    model.fit(X, y, epochs=100, batch_size= 32,  shuffle=False)

    X, y = create_window(df_test[headers], df_test['Carbohydrate intake'], utils.WINDOW_WIDTH_1H*2)
    y_pred = model(X)

    evaluate(y, y_pred, 15, 'CONV1', graph=True)

def lstm(df):
    df_train = df[:int(len(df)*0.8)]
    df_test = df[int(len(df)*0.8):]
    headers = [utils.ist_l, 'der1', 'grad1', 'grad2', 'grad3']

    X, y = create_window2(df_train[headers], df_train['cho2'], utils.WINDOW_WIDTH_1H*2, 12)

    model = tf.keras.Sequential()
    model.add(
          tf.keras.layers.LSTM(
              units=128,
              input_shape=[X.shape[1], X.shape[2]]
          )
    )
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    # model = tf.keras.models.Sequential([
	# 	# Shape [batch, time, features] => [batch, time, lstm_units]
	# 	tf.keras.layers.LSTM(32, return_sequences=True),
    #     tf.keras.layers.Conv1D(filters=3,
    #                        kernel_size=(utils.WINDOW_WIDTH_1H*2,),
    #                        activation='relu'),
    #     tf.keras.layers.Dense(units=128, activation='relu'),
	# 	# Shape => [batch, time, features]
	# 	tf.keras.layers.Dense(units=1)
	# ])

    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    model.fit(X, y, epochs=50, batch_size= 32,  shuffle=False)

    X, y = create_window(df_test[headers], df_test['Carbohydrate intake'], utils.WINDOW_WIDTH_1H*2)
    y_pred = model(X)
    # y_pred=y_pred[:,-1,0]

    evaluate(y, y_pred, 0.15, 'LSTM', graph=True)


def LDA(df):
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)

    headers = [utils.ist_l, 'quarter', 'weekday', 'der1', 'grad1', 'grad2', 'grad3']
    df['product'] = np.ones(len(df))
    for i, col in enumerate(headers):
        df['product'] = df['product'] * df[col]

    df_train = df[:int(len(df)*0.8)]
    df_eval = df[int(len(df)*0.8):]
    
    
    X = window_stack(df_train[[utils.ist_l]])
    y = df_train['cho2_b'][11:]
    lda.fit(X, y)
    qda.fit(X, y)

    X = window_stack(df_eval[[utils.ist_l]])
    y = df_eval['cho2_b'][11:]
    y_pred=lda.predict(X)
    evaluate(y, y_pred, 0, 'LDA window')
    y_pred=qda.predict(X)
    evaluate(y, y_pred, 0, 'QDA window')


    headers = [utils.ist_l, 'hour', 'quarter', 'weekday', 'der1', 'grad1', 'grad2', 'grad3', 'Steps']
    lda.fit(df_train[headers], df_train['cho2_b'])
    qda.fit(df_train[headers], df_train['cho2_b'])
    
    y_pred=lda.predict(df_eval[headers])
    evaluate(df_eval['cho2_b'], y_pred, 0, 'LDA multiple values')
    y_pred=qda.predict(df_eval[headers])
    evaluate(df_eval['cho2_b'], y_pred, 0, 'QDA multiple values')


    X = window_stack(df_train[['product']])
    y = df_train['cho2_b'][11:]
    lda.fit(X, y)
    qda.fit(X, y)

    X = window_stack(df_eval[['product']])
    y = df_eval['cho2_b'][11:]
    y_pred=lda.predict(X)
    evaluate(y, y_pred, 0, 'LDA window multiple values')
    y_pred=qda.predict(X)
    evaluate(y, y_pred, 0, 'QDA window multiple values')
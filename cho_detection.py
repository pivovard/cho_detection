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
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.title('FN')
    plt.scatter(range(end-begin), FN[begin:end], s=6)
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.title('FP')
    plt.scatter(range(end-begin), FP[begin:end], s=6)
    plt.legend()


def evaluate(y_label, y_pred, method='', graph=False):
    n = len(y_label)
    TP = np.zeros(n)
    FN = np.zeros(n)
    FP = np.zeros(n)
    delay = 0
    n_elements = 24 #2h
    y_elements = []

    for i, y in enumerate(y_label):
        if y > 0:
            if n - i >= n_elements:
                y_elements = y_pred[i:i+n_elements]
            else:
                y_elements = y_pred[i:]
            if np.any(y_elements > 0.5):
                TP[i] = True
            else:
                FN[i] = True
        elif y_pred[i] > 0.5:
            if i >= n_elements:
                y_elements = y_label[i-n_elements:i]
            else:
                y_elements = y_label[:i]
            if not np.any(y_elements > 0):
                FP[i] = True
    
    print(method)
    print(f'TP: {np.count_nonzero(TP)}')
    print(f'FN: {np.count_nonzero(FN)}')
    print(f'FP: {np.count_nonzero(FP)}')
    S= np.count_nonzero(TP)/(np.count_nonzero(TP)+np.count_nonzero(FN))
    print(f'S={S*100}%')

    if graph:
        plot_eval(y_label, y_pred, TP, FN, FP, title=f'{method}: All predicted values')
        plot_eval(y_label, y_pred, TP, FN, FP, end=utils.WINDOW_WIDTH_24H*2, title=f'{method}: 48h predicted')

def create_window(df, y_label, width):
    #shape (number of windows, window width, columns count)
    X = np.empty((len(df)-width, width, len(df.columns)))
    y = y_label[width:]
    for i in range(len(df)-width):
        X[i] = df[i:i+width]
    return X, y

def create_window2(df, y_label, width):
    #shape (number of windows, window width, columns count)
    X = np.empty((len(df)-width, width, len(df.columns)))
    y = y_label[int(width/2):-int(width/2)]
    for i in range(len(df)-width):
        X[i] = df[i:i+width]
    return X, y

def window_stack(a, stepsize=1, width=24):
    n = a.shape[0]
    return np.hstack( a[i:1+n+i-width:stepsize] for i in range(0,width) )

def dense(df):
    df_train = df[:int(len(df)*0.8)]
    df_test = df[int(len(df)*0.8):]
    headers = [utils.ist_l, 'hour', 'weekday', 'grad1', 'grad2', 'grad3']

    X, y = create_window2(df_train[headers], df_train['cho2_b'], utils.WINDOW_WIDTH_1H*2)

    model = tf.keras.Sequential([
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    model.fit(X, y, epochs=200, batch_size= 32,  shuffle=False)

    X, y = create_window(df_test[headers], df_test['cho_b'], utils.WINDOW_WIDTH_1H*2)
    y_pred = model(X)

    evaluate(y, y_pred, 'Dense', graph=True)


def conv(df):
    df_train = df[:int(len(df)*0.8)]
    df_test = df[int(len(df)*0.8):]
    headers = [utils.ist_l, 'quater', 'weekday', 'grad1', 'grad2', 'grad3']

    X, y = create_window2(df_train[headers], df_train['cho2'], utils.WINDOW_WIDTH_1H*2)

    model = tf.keras.models.Sequential([
		tf.keras.layers.Conv1D(filters=utils.WINDOW_WIDTH_1H*2,
                           kernel_size=(utils.WINDOW_WIDTH_1H*2,),
                           activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1)
	])

    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    model.fit(X, y, epochs=50, batch_size= 32,  shuffle=False)

    X, y = create_window(df_test[headers], df_test['cho2'], utils.WINDOW_WIDTH_1H*2)
    y_pred = model(X)

    evaluate(y, y_pred, 'CONV1', graph=True)

def lstm(df):
    df_train = df[:int(len(df)*0.8)]
    df_test = df[int(len(df)*0.8):]
    headers = [utils.ist_l, 'weekday', 'grad1', 'grad2', 'grad3']

    X, y = create_window(df_train[headers], df_train['cho2'], utils.WINDOW_WIDTH_1H*2)

    # enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # enc = enc.fit(y)
    # y = enc.transform(y)

    # model = tf.keras.Sequential()
    # model.add(
    #     tf.keras.layers.Bidirectional(
    #       tf.keras.layers.LSTM(
    #           units=128,
    #           input_shape=[X.shape[1], X.shape[2]]
    #       )
    #     )
    # )
    # model.add(tf.keras.layers.Dropout(rate=0.5))
    # model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    # model.add(tf.keras.layers.Dense(1, activation='softmax'))

    model = tf.keras.models.Sequential([
		# Shape [batch, time, features] => [batch, time, lstm_units]
		tf.keras.layers.LSTM(32, return_sequences=True),
		# Shape => [batch, time, features]
		tf.keras.layers.Dense(units=1)
	])

    print(X.shape)
    print(y.shape)

    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    model.fit(X, y, epochs=50, batch_size= 32,  shuffle=False)

    # lstm_model.evaluate(val_ds)
    # predict = lstm_model([np.array(test_df[0:32]),np.array(test_df[0:32])])


def LDA(df):
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)

    df_train = df[:int(len(df)*0.8)]
    df_eval = df[int(len(df)*0.8):]
    headers = [utils.ist_l, 'weekday', 'grad1', 'grad2', 'grad3']
    
    X2 = window_stack(df_train[[utils.ist_l]])
    y2 = y = df_train['cho2_b'][23:]
    print(X2.shape)
    lda.fit(X2, y2)
    qda.fit(X2, y2)

    X2 = window_stack(df_eval[[utils.ist_l]])
    y2 = y = df_eval['cho2_b'][23:]
    y_pred=lda.predict(X2)
    evaluate(y2, y_pred, 'LDA window')
    y_pred=qda.predict(X2)
    evaluate(y2, y_pred, 'QDA window')

    lda.fit(df_train[[utils.ist_l, 'weekday', 'grad1', 'grad2', 'grad3']], df_train['cho2_b'])
    qda.fit(df_train[[utils.ist_l, 'weekday', 'grad1', 'grad2', 'grad3']], df_train['cho2_b'])
    
    y_pred=lda.predict(df_eval[[utils.ist_l, 'weekday', 'grad1', 'grad2', 'grad3']])
    evaluate(df_eval['cho2_b'], y_pred, 'LDA value')
    y_pred=qda.predict(df_eval[[utils.ist_l, 'weekday', 'grad1', 'grad2', 'grad3']])
    evaluate(df_eval['cho2_b'], y_pred, 'QDA value')
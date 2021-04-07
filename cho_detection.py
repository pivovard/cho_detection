import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

from scipy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from datetime import timedelta
import utils

class ChoDetector():
    def __init__(self, df, graph):
        self.df_train = df[:int(len(df)*0.8)]
        self.df_test = df[int(len(df)*0.8):]
        self.graph = graph

def plot_eval(self, y_label, y_pred, TP, FN, FP, begin=0, end=0, title=''):
    if end==0:
        end=len(y_label)

    fig = plt.figure(figsize=(12, 8))
    fig.canvas.set_window_title(title)
    fig.suptitle(title)

    datetime = self.df_test['datetime'][begin:end]

    plt.subplot(4, 1, 1)
    plt.title('Predicted CHO')
    plt.scatter(datetime, y_pred[begin:end], label='predicted', s=6)
    plt.scatter(datetime, y_label[begin:end], label='label', s=6)
    # plt.plot(range(end-begin), self.df_test[begin:end])
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.title('TP')
    plt.scatter(datetime, TP[begin:end], s=6)

    plt.subplot(4, 1, 3)
    plt.title('FN')
    plt.scatter(datetime, FN[begin:end], s=6)

    plt.subplot(4, 1, 4)
    plt.title('FP')
    plt.scatter(datetime, FP[begin:end], s=6)

ChoDetector.plot_eval = plot_eval

def evaluate(self, y_label, y_pred, treshold=0, method=''):
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
                y_elements = y_pred[i-2:i+w]
            else:
                y_elements = y_pred[i-2:]
            if np.any(y_elements > treshold):
                TP[i] = True
            else:
                FN[i] = True
        elif y_pred[i] > treshold:
            if i >= w:
                y_elements = y_label[i-w:i]
            else:
                y_elements = y_label[:i]
            if not np.any(y_elements > 0) and np.all(FP[i-w:i]==False):
                FP[i] = True
    
    print(method)
    print(f'TP: {np.count_nonzero(TP)}')
    print(f'FN: {np.count_nonzero(FN)}')
    print(f'FP: {np.count_nonzero(FP)}')
    S= np.count_nonzero(TP)/(np.count_nonzero(TP)+np.count_nonzero(FN))
    print(f'S={S*100}%')

    if self.graph:
        self.plot_eval(y_label, y_pred, TP, FN, FP, title=f'{method}: All predicted values')
        # plot_eval(y_label, y_pred, TP, FN, FP, end=utils.WINDOW_WIDTH_24H*2, title=f'{method}: 48h predicted')

ChoDetector.evaluate = evaluate

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

def lstm(self):
    headers = ['ist', 'd1']

    X, y = create_window2(self.df_train[headers], self.df_train['cho2'], utils.WINDOW_WIDTH_1H*2, 6)

    model = tf.keras.Sequential()
    model.add(
          tf.keras.layers.LSTM(
              units=128,
              input_shape=[X.shape[1], X.shape[2]]
          )
    )
    #model.add(tf.keras.layers.Dropout(rate=0.5))
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
    model.fit(X, y, epochs=50, batch_size= 96,  shuffle=False)

    X, y = create_window(self.df_test[headers], self.df_test['Carbohydrate intake'], utils.WINDOW_WIDTH_1H*2)
    y_pred = model(X)
    # y_pred=y_pred[:,-1,0]

    self.evaluate(y, y_pred, 0.15, 'LSTM')

ChoDetector.lstm = lstm

def window_stack(a, stepsize=1, width=12):
    n = a.shape[0]
    return np.hstack( a[i:1+n+i-width:stepsize] for i in range(0,width) )

def replace_nan(df):
    df_clean = df.fillna(0)
    return df_clean

def lda(self):
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    
    df_train = replace_nan(self.df_train)
    df_test = replace_nan(self.df_test)

    X = window_stack(df_train[['d1']], width=24)
    y = df_train['cho2_b'][23:]
    lda.fit(X, y)
    qda.fit(X, y)

    X = window_stack(df_test[['d1']], width=24)
    y = df_test['cho_b'][23:]
    y_pred=lda.predict(X)
    self.evaluate(y, y_pred, 0, 'LDA window')
    y_pred=qda.predict(X)
    self.evaluate(y, y_pred, 0, 'QDA window')


    headers = ['ist', 'hour', 'quarter', 'weekday', 'd1', 'd2', 'd3', 'Steps']
    lda.fit(df_train[headers], df_train['cho2_b'])
    qda.fit(df_train[headers], df_train['cho2_b'])
    
    y_pred=lda.predict(df_test[headers])
    self.evaluate(df_test['cho_b'], y_pred, 0, 'LDA multiple values')
    y_pred=qda.predict(df_test[headers])
    self.evaluate(df_test['cho_b'], y_pred, 0, 'QDA multiple values')


    headers = ['ist', 'quarter', 'weekday', 'd1', 'd2', 'd3']
    df_train['product'] = np.ones(len(df_train))
    df_test['product'] = np.ones(len(df_test))
    for i, col in enumerate(headers):
        df_train['product'] = df_train['product'] * df_train[col]    
        df_test['product'] = df_test['product'] * df_test[col]    

    X = window_stack(df_train[['product']])
    y = self.df_train['cho2_b'][11:]
    lda.fit(X, y)
    qda.fit(X, y)

    X = window_stack(df_test[['product']])
    y = df_test['cho_b'][11:]
    y_pred=lda.predict(X)
    self.evaluate(y, y_pred, 0, 'LDA window multiple values')
    y_pred=qda.predict(X)
    self.evaluate(y, y_pred, 0, 'QDA window multiple values')

ChoDetector.lda = lda

def treshold_akima(self):
    d1_max = self.df_train['d1'].max()
    d2_max = self.df_train['d2'].max()
    d1t = [0.2*d1_max, 0.3*d1_max, 0.4*d1_max, 0.5*d1_max, 0.6*d1_max, 0.7*d1_max]
    d2t = [0.2*d2_max, 0.4*d2_max, 0.6*d2_max, 0.8*d2_max, 0.8*d2_max]

    datetime = self.df_test['datetime']

    fig = plt.figure(figsize=(12, 8))
    fig.canvas.set_window_title("d1")
    fig.suptitle("d1")

    plt.subplot(len(d1t)+2, 1, 1)
    plt.plot(datetime, self.df_test['d1'], label='d1')
    plt.legend()

    flags = np.zeros((len(d1t),len(self.df_test)))
    for i, val in enumerate(d1t):
        flags[i] = self.df_test['d1'] > val
        plt.subplot(len(d1t)+2, 1, i+2)
        plt.scatter(datetime, flags[i] + 0.5, label=f'{i}', s=5)
        plt.scatter(datetime, self.df_test['cho_b'], label='cho', s=5)
        plt.legend()

    #evaluate
    prc = np.sum(flags, axis=0)/len(d1t)
    plt.subplot(len(d1t)+2, 1, len(d1t)+2)
    plt.plot(datetime, prc, label='%')
    plt.scatter(datetime, self.df_test['cho_b'], label='cho', s=3)

    self.evaluate(self.df_test['cho_b'], prc, treshold=0.2, method='treshold')

ChoDetector.treshold_akima = treshold_akima

def treshold_manual(self, df=None):
    d1t = [0, 0.5, 1.25, 1.8]
    d1t = [0, 0.005, 0.0125, 0.018]
    d2t = [0, 0.005, 0.0125, 0.018]
    weight = [1,1.5,2.25,3]

    if df is None:
        df = self.df_test.reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    
    datetime = df['datetime']

    flags1 = np.zeros((len(d1t),len(df)))
    flags2 = np.zeros((len(d1t),len(df)))
    for i, val in enumerate(d1t):
        flags1[i] = (df['d1'] >= d1t[i]) * weight[i]
        flags2[i] = (df['d2'] >= d2t[i]) * weight[i]

    flags1 = np.max(flags1, axis=0)
    flags2 = np.max(flags2, axis=0)
    
    cross = np.zeros(len(df))
    for i in range(1, len(df)):
        if ((df.loc[i-1,'d1'] > df.loc[i,'d1'] and df.loc[i-1,'d2'] < df.loc[i,'d2']) or
            (df.loc[i-1,'d1'] < df.loc[i,'d1'] and df.loc[i-1,'d2'] > df.loc[i,'d2'])):
            cross[i] = 3
    
    for i in range(12, len(df)):
        for j in range(1, 13):
            if flags1[i] >= 2 and flags1[i-j] >= 2+0.2*j:
                flags1[i] = flags1[i] + 0.1*j

        # if flags1[i] >= 2 and flags1[i-2] >= 3:
        #     flags1[i] = flags1[i]+1
    #     if flags1[i] == 1.5 and s <= 12:
    #         flags1[i] = 3.5
    #     if flags1[i] == 2.25 and s <= 12:
    #         flags1[i] = 4
    #     if s >= 1 and s <= 5 and np.max(flags1[i-3:i]) == 1:
    #         flags1[i] = 4


    detected=np.zeros(len(df))
    for i, val in enumerate(flags1):
        if val >= 2:
            detected[i] = val
        else:
            detected[i]=None

    fig = plt.figure(figsize=(12, 8))
    fig.canvas.set_window_title("d1")
    fig.suptitle("Pacient 575")

    plt.subplot(3, 1, 1)
    plt.plot(datetime, df['Interstitial glucose'], label='ist')
    plt.plot(datetime, df['Heartbeat']*0.2, label='heartbeat *0.2')
    plt.scatter(datetime, df['cho']*0.2, label='cho *0.2', s=10, c='g')
    plt.scatter(datetime, df[utils.phy_l], label='activity', s=10, c='b', marker='^')
    plt.scatter(datetime, detected, label='detected cho', s=10, c='r', marker='*')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(datetime, np.full(len(df), 0.018), label='treshold 0.018')
    plt.plot(datetime, np.full(len(df), 0.0125), label='treshold 0.0125')
    plt.plot(datetime, df['d1'], label='d1')
    # plt.plot(datetime, df['d2'], label='d2')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(datetime, flags1, label='activation')
    # plt.plot(datetime, df['Heartbeat'], label='heartbeat')
    
    plt.legend()

ChoDetector.treshold_manual = treshold_manual
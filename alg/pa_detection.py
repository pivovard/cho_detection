"""
This scripts handles physical activity detection
- Calculation features of given columns (mean, median, std, quantiles) and export to csv
- Machine learning algorithms test

@author Bc. David Pivovar
"""

from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import statistics
import sweetviz

import subprocess
from datetime import timedelta
import alg.utils as utils

## plot data and features
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

## Calculate features of given columns (mean, median, std, quantiles)
def get_features(patientID, df, headers, test=''):
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

    data.to_csv(f'data/{patientID}-pa{test}.csv', index=False, sep=';')

    plot_pa(data, headers)
    # report = sweetviz.analyze(data)
    # report.show_html()

    return data

## Export features to csv file (without headers)
def export_pa(df, headers, patientID):
    df = pd.read_csv(f'data/{patientID}-pa.csv', sep=';')
    data = pd.DataFrame()
    data['pa'] = df['pa2'].apply(lambda val : (int)(cond(val)) )

    for i, val in enumerate(headers):
        # data[f'{val}']=df[f'{val}']
        data[f'{val} mean']=df[f'{val} mean']
        data[f'{val} median']=df[f'{val} median']
        data[f'{val} std']=df[f'{val} std']
        data[f'{val} kvartil']=df[f'{val} kvartil']
    
    data.fillna(0, inplace=True)    
    print(data.head())
    data.to_csv(f'data/{patientID}-pa-export.csv', sep=',', index=False, header=False)

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.discriminant_analysis
import sklearn.neighbors
import sklearn.tree
import sklearn.svm
import sklearn.neural_network

def cond(val):
    if val > 0:
        return True
    else:
        return False

## Machine learning algorithms test
def ML(patientID, headers, scale=''):
    df = pd.read_csv(f'data/{patientID}-pa.csv', sep=';')
    df=df.fillna(0)
    df_test = pd.read_csv(f'data/{patientID}-pa-test.csv', sep=';')
    df_test=df_test.fillna(0)

    X=pd.DataFrame()
    X_train=pd.DataFrame()
    X_test=pd.DataFrame()
    for i, val in enumerate(headers):
        X_train[f'{val} mean']=df[f'{val} mean']
        X_train[f'{val} median']=df[f'{val} median']
        X_train[f'{val} std']=df[f'{val} std']
        X_train[f'{val} kvartil']=df[f'{val} kvartil']

        X_test[f'{val} mean']=df_test[f'{val} mean']
        X_test[f'{val} median']=df_test[f'{val} median']
        X_test[f'{val} std']=df_test[f'{val} std']
        X_test[f'{val} kvartil']=df_test[f'{val} kvartil']

    y_train= df['pa2'].apply(lambda val : cond(val) )
    y_test= df_test['pa']

    if scale == 'std':
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    elif scale == 'minmax':
        scaler = preprocessing.MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    print('\nBayes')
    gnb = sklearn.naive_bayes.GaussianNB()    
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    utils.evaluate(y_test, y_pred, 0)

    print('\nLDA')
    gnb = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    utils.evaluate(y_test, y_pred, 0)

    print('\nGDA')
    gnb = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()    
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    utils.evaluate(y_test, y_pred, 0)

    print('\nkNN')
    gnb = sklearn.neighbors.KNeighborsClassifier()    
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    utils.evaluate(y_test, y_pred, 0)

    print('\nTree')
    gnb = sklearn.tree.DecisionTreeClassifier()    
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    utils.evaluate(y_test, y_pred, 0)

    print('\nSVM')
    gnb = sklearn.svm.SVC()    
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    utils.evaluate(y_test, y_pred, 0)

    print('\nLogisticRegression')
    gnb = sklearn.linear_model.LogisticRegression(max_iter=500)
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    utils.evaluate(y_test, y_pred, 0)

    print('\nLogisticRegressionCV')
    gnb = sklearn.linear_model.LogisticRegressionCV(max_iter=500)    
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    utils.evaluate(y_test, y_pred, 0)

    print('\nPerceptron')
    gnb = sklearn.linear_model.Perceptron()    
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    utils.evaluate(y_test, y_pred, 0)

    print('\nMLP')
    gnb = sklearn.neural_network.MLPClassifier()    
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    utils.evaluate(y_test, y_pred, 0)
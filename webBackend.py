import os
import warnings
import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from pmdarima.arima import auto_arima

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sns.set(rc={'figure.figsize':(11, 4)})

#Chacking statinarity automatically
def checkStationarity(df):
    df = df['Siste']
    Series = pd.Series(df)
    nrOfDiff= 0
    result = adfuller(Series)
    k= 0
    for key, value in result[4].items():
        if (k == 0):
            myvalue = value
            break
        k += 1

    adfStat = result[0]

    if(adfStat< myvalue*3):
        return nrOfDiff
    else:
        while(nrOfDiff <=2):
            if(nrOfDiff == 2):
                return nrOfDiff
            nrOfDiff+=1
            df = df.diff().dropna()
            Series= pd.Series(df)
            result = adfuller(Series)
            adfStat = result[0]

            for key, value in result[4].items():
                myvalue = value
                break

            if(adfStat< myvalue):

                return nrOfDiff

#finding otimal order with autoarima and AIC
def findOrder(df):
    nrofdiff = checkStationarity(df)
    df = df['Siste']

    print("starting")
    #model_auto = auto_arima(df, test='adf',d=nrofdiff, trace=True, error_action='ignore',start_p=2, start_P=2, start_Q=2,start_q=2, max_p=5, max_q=5,
                            #suppress_warnings=True, stepwise=False, seasonal=False, max_order=10)

    global order
    order=(2,1,2)

    #order = model_auto.order
    print(order)
    #print(model_auto.summary())



#reading files that user want to look at
def testData(file):
    df = pd.read_csv(file, sep=';')
    df = df.dropna()
    index = 0
    for column in df.columns:
        index+= 1
        if(index == 1):
            df[column] = pd.to_datetime(df[column], format='%d.%m.%y')
            df.sort_values(by=[column])
            df = df.iloc[::-1]
            df = df.set_index(column)
        else:
            if(' ' in df[column][3]):
                df[column] = df[column].str.replace(' ', '').astype(np.float64)
            else:
                df[column] = df[column].str.replace(',', '.').astype(np.float64)
    return df

#creating arima forecast
def websiteArima(df, step):
    df = df['Siste']
    print(order)
    prosentDF = 0.9
    n = int(len(df) * prosentDF)
    train = df[:n]
    test = df[n:]

    plotDf = df[n - step:n + step]

    model = ARIMA(train, order)

    result = model.fit(full_output=True)

    fc, se, conf = result.forecast(step)

    new_graph_name = checkWebResults(fc, conf, test, step, plotDf)

    return new_graph_name

#plotting data
def checkWebResults(fc, conf, test, step, plotDf):

    #checking results

    fc = pd.Series(fc, index=test[:step].index)
    lower = pd.Series(conf[:, 0], index=test[:step].index)
    upper = pd.Series(conf[:, 1], index=test[:step].index)

    plt.figure(figsize=(16,8))
    plt.plot(plotDf, label= "actual")
    plt.plot(fc, label= "forecast")
    plt.fill_between(lower.index, lower, upper, color= 'k', alpha= 0.1)
    plt.title("Forecast")
    plt.legend(loc="upper left")

    new_graph_name = "graph_" + str(time.time()) + '.png'

    for filename in os.listdir('static/'):
        if filename.startswith('graph_'):  # not to remove other images
            os.remove('static/' + filename)

    plt.savefig(ROOT_DIR +'\\static\\' + new_graph_name)

    #plt.savefig(ROOT_DIR + '\\static\\checkresults1.png')

    return new_graph_name

#creating prediction
def createPredictions(df, step):
    df = df['Siste']

    model = ARIMA(df, order)

    result = model.fit(full_output=True)

    fc, se, conf = result.forecast(step)

    df= df.tail(step*2)

    lastnum = df.tail(1)[0]

    newArray=[]
    newArray.append(lastnum)
    for i in fc:
        newArray.append(i)

    fc= newArray

    newname = makePredictionPlot(fc, conf, df, step)

    return newname

#plotting prediction
def makePredictionPlot(fc, conf, df, step):

    dateIndex= pd.date_range(df.tail(1).index[0], periods=step+1, freq='D')
    confDateIndex = dateIndex[1:]

    fc = pd.Series(fc, index=dateIndex)
    lower = pd.Series(conf[:, 0], index=confDateIndex)
    upper = pd.Series(conf[:, 1], index=confDateIndex)


    plt.figure(figsize=(16,8))
    plt.plot(df, label= "actual")
    plt.plot(fc, label= "forecast")
    plt.fill_between(lower.index, lower, upper, color= 'k', alpha= 0.1)
    plt.title("Forecast")
    plt.legend(loc="upper left")

    new_graph_name = "pred_" + str(time.time()) + '.png'

    for filename in os.listdir('static/'):
        if filename.startswith('pred_'):  # not to remove other images
            os.remove('static/' + filename)

    plt.savefig(ROOT_DIR + '\\static\\' + new_graph_name)

    return new_graph_name

#-------------------------------------------rest of the code is for creating ACF and PACF plots
def find_nondiff_p(df):

    df = df['Siste']

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (16,4))

    ax1.plot(df)
    ax2.set_title("Difference once")
    plot_pacf(df, ax= ax2)

    new_graph_name = "nondiffP_" + str(time.time()) + '.png'

    for filename in os.listdir('static/'):
        if filename.startswith('nondiffP_'):  # not to remove other images
            os.remove('static/' + filename)

    plt.savefig(ROOT_DIR + '\\static\\' + new_graph_name)

    return new_graph_name
    #we can see that PACF lag 7 is significant as it's above significance line.

def find_onediff_p(df):

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (16,4))

    ax1.plot(df)
    ax2.set_title("Difference once")
    plot_pacf(df, ax= ax2)

    new_graph_name = "onediffP_" + str(time.time()) + '.png'

    for filename in os.listdir('static/'):
        if filename.startswith('onediffP_'):  # not to remove other images
            os.remove('static/' + filename)

    plt.savefig(ROOT_DIR + '\\static\\' + new_graph_name)

    return new_graph_name

    #we can see that PACF lag 7 is significant as it's above significance line.




def find_nondiff_q(df):

    df = df['Siste']

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (16,4))

    ax1.plot(df)
    ax2.set_title("Difference once")
    plot_acf(df, ax= ax2)

    new_graph_name = "nondiffQ_" + str(time.time()) + '.png'

    for filename in os.listdir('static/'):
        if filename.startswith('nondiffQ_'):  # not to remove other images
            os.remove('static/' + filename)

    plt.savefig(ROOT_DIR + '\\static\\' + new_graph_name)

    return new_graph_name
    #we can see that PACF lag 7 is significant as it's above significance line.


def find_onediff_q(df):

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (16,4))

    ax1.plot(df)
    ax2.set_title("Difference once")
    plot_acf(df, ax= ax2)

    new_graph_name = "onediffQ_" + str(time.time()) + '.png'

    for filename in os.listdir('static/'):
        if filename.startswith('onediffQ_'):  # not to remove other images
            os.remove('static/' + filename)

    plt.savefig(ROOT_DIR + '\\static\\' + new_graph_name)

    return new_graph_name

    #we can see that PACF lag 7 is significant as it's above significance line.


def return_Stationary(df):

    df_final = pd.Series(df['Siste'])

    diff = df_final.diff().dropna()

    return diff

def create_autocorr_p(df):
    nondiff = find_nondiff_p(df)
    newdf = return_Stationary(df)
    onediff = find_onediff_p(newdf)

    return [nondiff,onediff]

def create_autocorr_q(df):
    nondiff = find_nondiff_q(df)
    newdf = return_Stationary(df)
    onediff = find_onediff_q(newdf)
    return[nondiff, onediff]


import os
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

from sklearn.metrics import mean_squared_error

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sns.set(rc={'figure.figsize':(11, 4)})


def createDf():
    df = pd.read_csv(ROOT_DIR + '\\eqData.csv', sep = ';')
    return df

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


def websiteArima(df):
    df = df['Siste']

    prosentDF = 0.5
    n = int(len(df) * prosentDF)
    train = df[:n]
    test = df[n:]
    step = 20
    print(step)

    plotDf = df[n - step:n + step]

    print(plotDf)

    model = ARIMA(train, (7, 1, 1))

    result = model.fit(full_output=True)

    fc, se, conf = result.forecast(step)

    print(fc)

    checkWebResults(fc, conf, test, step, plotDf)


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
    plt.savefig(ROOT_DIR + '\\static\\newbie.png')


def updatedDataSet():
    df = createDf()
    df["EQNR"] = pd.to_datetime(df['EQNR'], format='%d.%m.%Y')

    #lastdf= df[["EQNR", "Siste"]]

    df.sort_values(by=['EQNR'])

    df = df.iloc[::-1]

    df = df.set_index('EQNR')

    df['Siste'] = df['Siste'].str.replace(',', '.').astype(np.float64)
    df['Kjøper'] = df['Kjøper'].str.replace(',', '.').astype(np.float64)
    df['Selger'] = df['Selger'].str.replace(',', '.').astype(np.float64)
    df['Høy'] = df['Høy'].str.replace(',', '.').astype(np.float64)
    df['Lav'] = df['Lav'].str.replace(',', '.').astype(np.float64)
    df['VWAP'] = df['VWAP'].str.replace(',', '.').astype(np.float64)
    df['Totalt omsatt (NOK)'] = df['Totalt omsatt (NOK)'].str.replace(' ', '').astype(np.float64)
    df['Totalt antall aksjer omsatt'] = df['Totalt antall aksjer omsatt'].str.replace(' ', '').astype(np.float64)
    df['Antall off. handler'] = df['Antall off. handler'].str.replace(' ', '').astype(np.float64)
    df['Antall handler totalt'] = df['Antall handler totalt'].str.replace(' ', '').astype(np.float64)

    return df

def makePlot():
    updatedDataSet().plot()
    plt.show()


def createLagPlot():

    pd.plotting.lag_plot(updatedDataSet(), lag= 3)
    plt.title('EQNR Stock - Autocorrelation plot with lag = 3')
    plt.show()


def create_ac_plot(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    ax1.plot(df)
    ax1.set_title("Difference once")
    plot_acf(df, ax=ax2)
    plt.show()


#checking for stationarity

def check_stationarity(ts_data):

    # Rolling statistics

    roll_mean = ts_data.rolling(30).mean()
    roll_std = ts_data.rolling(5).std()

    # Plot rolling statistics
    fig = plt.figure(figsize=(20, 10))
    plt.subplot(211)
    plt.plot(ts_data, color='black', label='Original Data')
    plt.plot(roll_mean, color='red', label='Rolling Mean(30 days)')
    plt.legend()
    plt.show()
    plt.subplot(212)
    plt.plot(roll_std, color='green', label='Rolling Std Dev(5 days)')
    plt.legend()
    plt.show()

    #printing autocorrolation
    create_ac_plot(ts_data)

    # Dickey-Fuller test
    print('Dickey-Fuller test results\n')
    df_test = adfuller(ts_data, regresults=False)
    test_result = pd.Series(df_test[0:4], index=['Test Statistic', 'p-value', '# of lags', '# of obs'])
    print(test_result)
    for k, v in df_test[4].items():
        print('Critical value at %s: %1.5f' % (k, v))




def isMyDataStationary():
    df_final = pd.Series(updatedDataSet()['Siste'])

    #check_stationarity(df_final)

    diff = df_final.diff().dropna()

    check_stationarity(diff)


def return_Stationary():
    df_final = pd.Series(updatedDataSet()['Siste'])

    diff = df_final.diff().dropna()

    '''
    df_final = pd.Series(updatedDataSet()['Siste'])


    df_final_log = np.log(df_final)

    df_final_log_diff = df_final_log - df_final_log.shift(1)


    df_final_log_diff = df_final_log_diff.dropna()

    '''

    return diff


def find_p(df):

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (16,4))

    ax1.plot(df)
    ax2.set_title("Difference once")
    ax2.set_ylim(0,1)
    plot_pacf(df, ax= ax2)
    plt.show()

    #we can see that PACF lag 7 is significant as it's above significance line.

def find_q(df):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

    ax1.plot(df)
    ax2.set_title("Difference once")
    ax2.set_ylim(0, 1)
    plot_acf(df, ax=ax2)
    plt.show()
    # 7 is suitable

def implement_arima():

    df = updatedDataSet()['Siste']

    prosentDF = 0.5
    n = int(len(df) * prosentDF)
    train = df[:n]
    test = df[n:]


    step = 20

    plotDf = df[n - step:n + step]

    #series = return_Stationary()

    #pd.set_option('display.max_columns', None)  # or 1000

    model = ARIMA(train,(7, 1, 7))


    result = model.fit(full_output=True)

    fc, se, conf = result.forecast(step)

    #predictions = model.predict(model_fit.params)

    #summery = model_fit.summary()

    #model_fit.plot_predict(start=1, end= 100, dynamic=False)
    #plt.show()

    checkResults(fc, conf, test, step, plotDf)

def checkResults(fc, conf, test, step, plotDf):

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
    #plt.show()

if __name__ == '__main__':

    x=1
    print(x)
    #testData()
    '''    
    df = return_Stationary()
    find_q(df)

    #isMyDataStationary()

    '''
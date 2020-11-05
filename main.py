import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sns.set(rc={'figure.figsize':(11, 4)})


def createDf():
    df = pd.read_csv(ROOT_DIR + '\\eqData.csv', sep = ';')
    return df

def updatedDataSet():
    df = createDf()
    df["EQNR"] = pd.to_datetime(df['EQNR'], format='%d.%m.%Y')

    df = df.asfreq('D')


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

    # Dickey-Fuller test
    print('Dickey-Fuller test results\n')
    df_test = adfuller(ts_data, regresults=False)
    test_result = pd.Series(df_test[0:4], index=['Test Statistic', 'p-value', '# of lags', '# of obs'])
    print(test_result)
    for k, v in df_test[4].items():
        print('Critical value at %s: %1.5f' % (k, v))




def checking_stationarity():
    df_final = pd.Series(updatedDataSet()['Siste'])
    # check_stationarity(df_final)

    df_final_log = np.log(df_final)

    df_final_log.dropna(inplace=True)
    # check_stationarity(df_final_log)

    print(df_final_log.shift())
    df_final_log_diff = df_final_log - df_final_log.shift()
    df_final_log_diff.dropna(inplace=True)

    check_stationarity(df_final_log_diff)


def getStationary():
    df_final = pd.Series(updatedDataSet()['Siste'])

    df_final_log = np.log(df_final)

    df_final_log.dropna(inplace=True)

    df_final_log_diff = df_final_log - df_final_log.shift(1)


    df_final_log_diff = df_final_log_diff.dropna()

    return df_final_log_diff

def implement_arima():

    series = getStationary()

    series = series.asfreq('D')


    model = ARIMA(endog= series, order= (0, 1, 1))
    model.fit()

if __name__ == '__main__':
    implement_arima()
    #print(len(getStationary()))

import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

from pmdarima.arima import auto_arima

import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sns.set(rc={'figure.figsize':(11, 4)})


def createDf():
    df = pd.read_csv(ROOT_DIR + '\\eqData.csv', sep = ';')
    return df


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


def return_Stationary(df):


    df_final = pd.Series(df['Siste'])

    diff = df_final.diff().dropna()


    return diff


def find_p(df):
    plot_pacf(df)
    plt.xlabel('LAG')
    plt.ylabel('PACF')
    plt.savefig(ROOT_DIR + '\\images\\PACF.png')

    #we can see that PACF lag 7 is significant as it's above significance line.



def find_qq(df):
    plot_acf(df)
    plt.xlabel('LAG')
    plt.ylabel('ACF')
    plt.savefig(ROOT_DIR + '\\images\\ACF.png')

def find_q(df, fileName):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    ax1.plot(df)
    ax2.set_title("Difference once")
    plot_acf(df, ax=ax2)


    new_graph_name = "graph_" + str(time.time())
    print(new_graph_name)

    for filename in os.listdir('static/'):
        if filename.startswith('graph_'):  # not to remove other images
            os.remove('static/' + filename)

    plt.savefig(ROOT_DIR +'\\static\\' + new_graph_name +'.png')



    plt.savefig(ROOT_DIR + '\\static\\' + fileName + '.png')
    # 7 is suitable


from sklearn.metrics import mean_squared_error
from math import sqrt

def print_errors(actual, forecast):
    rms = sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE

    fcError = abs(actual-forecast).mean()

    print('-'*100)
    print('fcErros: ' + str(fcError))
    print('-'*100)
    print('rms: '+ str(rms))
    print('-'*100)
    print('mape: ' +str(mape))



def create_preplot(model_fit, df):
    a= model_fit.plot_predict(dynamic=False)
    print(type(a))
    a.to_csv(ROOT_DIR+ '\\pred.csv')
    plt.show()
    a = a[1:1252]
    df = df[2:1253]

    a.to_csv(ROOT_DIR +'\\Arima(6,1,7).csv')
    '''
    #print_errors(df, a)
    df.plot(label= 'actual')
    plt.plot(a, label= 'forecast')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('EQRN')
    plt.savefig(ROOT_DIR+ '\\images\\smallfcplot1.png')
    
    '''

def find_error_preday():
    df = updatedDataSet()['Siste']
    model = ARIMA(df, (6, 1, 7))
    result = model.fit(full_output=True)
    create_preplot(result, df)

def implement_arima():

    df = updatedDataSet()['Siste']

    prosentDF = 0.7
    n = int(len(df) * prosentDF)
    train = df[:n]
    test = df[n:]


    step = 5

    plotDf = df[n - step:n + step]


    model = ARIMA(train,(7, 1, 7))

    result = model.fit(full_output=True)

    #create_residuals(result)
    #print(result.summary())
    create_preplot(result, df)

    fc, se, conf = result.forecast(step)

    #checkResults(fc, conf, test, step, plotDf)

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
    plt.savefig(ROOT_DIR + '\\images\\plotTrain111.png')


def create_residuals(model_fit):
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(1, 2)
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.savefig(ROOT_DIR +'\\images\\resudual.png')

def create_series_plot(df):
    df.plot()
    plt.title('Closed stock prices Equinor')
    plt.xlabel('date')
    plt.ylabel('closed price')
    plt.savefig(ROOT_DIR + '\\images\\EQNR.png')



#------------------------------------------------------------------------------------

def makeAutoArima():
    df = updatedDataSet()['Siste']

    prosentDF = 0.7
    n = int(len(df) * prosentDF)
    train = df[:n]
    test = df[n:]

    step = 5

    plotDf = df[n - step:n + step]

    model_auto = auto_arima(train, trace=True, error_action='ignore', start_p=1,start_q=1,max_p=8,max_q=8,
              suppress_warnings=True,stepwise=False,seasonal=False, max_order=20)

    print(model_auto.order)
    print(model_auto)
    print(model_auto.summary())

def checkStationarity(df):
    df = df['Siste']
    Series = pd.Series(df)
    nrOfDiff= 0
    result = adfuller(Series)
    k= 0
    for key, value in result[4].items():
        if (k == 0):
            myvalue = value
            print('\t%s: %.3f' % (key, value))
            break
        k += 1

    adfStat = result[0]
    print(adfStat)
    print(myvalue)

    if(adfStat< myvalue):
        return nrOfDiff
    else:
        while(nrOfDiff <=2):
            nrOfDiff+=1
            df = df.diff().dropna()
            Series= pd.Series(df)
            result = adfuller(Series)
            adfStat = result[0]
            k= 0
            for key, value in result[4].items():
                if(k== 0):
                    myvalue = value
                    print('\t%s: %.3f' % (key, value))
                    break
                k += 1

            if(adfStat< myvalue):
                print(adfStat)
                print(myvalue)
                return nrOfDiff

if __name__ == '__main__':
    print(checkStationarity(updatedDataSet()))
    #makeAutoArima()
    #find_error_preday()
    #create_preplot()
    #implement_arima()

    '''
    diff = pd.Series(updatedDataSet()['Siste'])
    diff = diff.diff().dropna()
    find_qq(diff)
    '''
    #create_series_plot(diff)
    #create_ac_plot(df_final)
    #isMyDataStationary()

    '''    
    x=1
    print(x)
    df = updatedDataSet()['Siste']
    find_q(df, 'nondiff_p')

    '''

    #testData()

    '''    
    df = return_Stationary()
    find_q(df)
    #isMyDataStationary()
    
    '''

# Forecasting Stock Closing Price with Autoregressive Integrated Moving Average :chart_with_upwards_trend:

This program is made for predicting the future stock prices of Equinor.
 The program is calculated in main.py where the ARIMA model is integrated. 
 Website is also implemented in webBackend.py and app.py with flask. 
 The website only uses auto_arima for calculating the ARIMA model.

# Installation

Use the packages pip to install statsmodels and pmdarima

```bash
pip install statsmodels
pip install pmdarima
```

# Usage

```python
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima

if __name__ == '__main__':
    df = updatedDataSet() #creating dataset
    isMyDataStationary() # checking stationarity with dickey fuller test
    df = return_Stationary(df) # return nr. of stationarity
    find_p(df) #creating pac plot, optimal value = 7
    find_qq(df) #creating pacf plot, optimal value = 7
    implement_arima() #Use arima to calculate prediction with given values
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


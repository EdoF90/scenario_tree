import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

tickers = ["META", "GOOGL", "AMZN"]
train_data = yf.download(tickers, start='2023-01-01', end='2024-05-01')['Adj Close']
train_data = train_data.dropna()
test_data= yf.download(tickers, start='2024-05-01', end='2024-05-20')['Adj Close']
test_data = test_data.dropna()

forecasted_means = {}
confidence_intervals = {}
for ticker in train_data.columns:
    model = auto_arima(y=train_data[ticker], seasonal=False, stepwise=True)
    forecast = model.predict(n_periods=len(test_data))
    forecasted_means[ticker] = forecast
    # confidence_intervals[ticker] = conf_int
    print(model.summary())
    
forecast_index = pd.date_range(start=train_data.index[-1] + pd.Timedelta(days=1), periods=len(test_data), freq='B')
forecasted_means_df = pd.DataFrame(index=forecast_index, columns=tickers)
for ticker in tickers:
    forecasted_means_df[ticker] = forecasted_means[ticker].values
# confidence_intervals_df = {ticker: pd.DataFrame(conf_int, columns=['lower', 'upper'], index=forecasted_means_df.index) for ticker, conf_int in confidence_intervals.items()}
'''
for ticker in tickers:
    rmse = np.sqrt(mean_squared_error(test_data[ticker], forecasted_means_df[ticker]))
    print(f"RMSE for {ticker}: {rmse}")
'''
for ticker in tickers:
    plt.figure(figsize=(14, 7))
    plt.plot(test_data.index, test_data[ticker], label='Actual Prices', color='blue')
    plt.plot(forecasted_means_df.index, forecasted_means_df[ticker], label='Forecasted Prices', color='orange', linestyle='--')
    plt.title(f'{ticker} Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
'''
plt.fill_between(forecasted_means_df.index, 
                 confidence_intervals_df[ticker]['lower'], 
                 confidence_intervals_df[ticker]['upper'], 
                 color='gray', alpha=0.2, label='95% Confidence Interval')
'''


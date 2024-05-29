import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

tickers = ["META", "GOOGL", "AMZN"]
prices = yf.download(
        tickers,
        start= '2022-05-01',
        end= '2024-05-15'
        )['Adj Close']
#returns = data.pct_change().dropna()

# Normalizza i rendimenti
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Funzione per creare dataset con una finestra temporale
def create_dataset(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Creare il dataset
window_size = 60
X, y = create_dataset(scaled_prices, window_size)

# Divisione del dataset in training e test
train_size = int(len(X) * 0.95)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape per LSTM (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], len(tickers))
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], len(tickers))

# Costruire il modello LSTM
model = Sequential()
model.add(LSTM(units=250, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(units=175, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(units=len(tickers)))

# Compilare il modello
model.compile(optimizer='adam', loss='mean_squared_error')

# Addestrare il modello
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Fare previsioni
predicted_prices = model.predict(X_test)

# Invertire la normalizzazione delle previsioni
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.figure(figsize=(14,5))

# Prezzi reali (invertire la normalizzazione dei dati originali)
real_prices = scaler.inverse_transform(scaled_prices[train_size+window_size:])

# Prezzi reali per ogni ticker
for i, ticker in enumerate(tickers):
    plt.plot(prices.index[train_size+window_size:], real_prices[:, i], label=f'Prezzi Reali - {ticker}')
    plt.plot(prices.index[train_size+window_size:], predicted_prices[:, i], label=f'Prezzi Previsti - {ticker}')
    plt.title('Prezzi Azionari Reali vs Previsti')
    plt.xlabel('Data')
    plt.ylabel('Prezzo')
    plt.legend()
    plt.show()


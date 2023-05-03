import numpy as np
import yfinance as yf
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

# Pobierz historyczne ceny ropy naftowej WTI Crude Oil
data = yf.download("CL=F", start="2022-02-24", end="2023-02-24")

# Wyświetl dane
print(data)

# Wybierz kolumnę z cenami ropy
price = data['Low'].values

# Normalizuj ceny
price = price.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
price = scaler.fit_transform(price)

# Podziel dane na zestawy uczące i testowe
train_size = int(len(price) * 0.67)
test_size = len(price) - train_size
train_data, test_data = price[0:train_size, :], price[train_size:len(price), :]
train_data = train_data.squeeze()
test_data = test_data.squeeze()


# Przygotuj dane dla sieci LSTM
def prepare_data(data_, look_back_=1):
    x, y = [], []
    for i in range(len(data_) - look_back_ - 1):
        a = data_[i:(i + look_back_)]
        x.append(a)
        y.append(data_[i + look_back_])
    return np.array(x), np.array(y)


look_back = 1
train_X, train_Y = prepare_data(train_data, look_back)
test_X, test_Y = prepare_data(test_data, look_back)

# Stwórz sieć LSTM
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=Adam())

# Trenuj sieć LSTM
model.fit(train_X, train_Y, epochs=50, batch_size=1)

# Dokonaj predykcji cen ropy naftowej
predicted_price = model.predict(test_X)
predicted_price = scaler.inverse_transform(predicted_price)

# Oceń wyniki predykcji
test_Y = scaler.inverse_transform([test_Y])
mse = mean_squared_error(test_Y[0], predicted_price[:, 0])
print('MSE:', mse)
mape = mean_absolute_percentage_error(test_Y[0], predicted_price[:, 0]) * 100
print(f'MAPE: {mape:.2f}%')

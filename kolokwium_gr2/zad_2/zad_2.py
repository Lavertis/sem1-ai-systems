import numpy as np
import pandas as pd
from keras import Input, Model
from keras.layers import LSTM, Dense, TimeDistributed, GRU
from keras.losses import MeanAbsolutePercentageError
from keras.optimizers.optimizer_v2.rmsprop import RMSProp
from keras.utils import plot_model
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error


# Przeprowadź predykcję szeregu czasowego z pliku .csv wykorzystując 2 sztuczne
# sieci neuronowe o różnej strukturze, gdzie jedna zawiera warstwę LSTM, a druga
# GRU.
# a. Przygotowanie danych: przekształć dane tak, aby możliwe było
# prognozowanie wartości szeregu czasowego na podstawie liczby obserwacji
# równej liczbie liter w Twoim imieniu. Prognoza ma być możliwa na chwilę
# t+1 oraz t+k, gdzie k jest liczbą spółgłosek w imieniu. np. Karol (5 liter i 3
# spółgłoski) przygotowuje zbiory danych pozwalające na predykcję na
# podstawie 5 poprzednich obserwacji wartości kolejnej (t+1) oraz wartości w
# chwili (t+3).
# b. Stwórz 2 sztuczne sieci neuronowe zawierające warstwy LSTM i użyj je do
# predykcji wartości szeregu na zbiorach z podpunktu a. (w sumie 4 próby
# uczenia sieci, podział zbioru na treningowy i testowy 90% do 10%). W
# raporcie podaj struktury użytych sieci, ich wyniki nauki i predykcji ocenione
# metryką MAPE oraz predykcje w formie wykresów.


def make_dataset(dataset, obs_length, pred_offset):
    x_length = len(dataset) - obs_length - pred_offset + 1
    x_ = np.zeros((x_length, obs_length, 1))
    y_ = np.zeros((x_length, 1))

    for i in range(x_length):
        x_[i] = dataset[i:i + obs_length]
        y_[i] = dataset[i + obs_length + pred_offset - 1]

    return x_, y_


def normalize_array(train, test, axis):
    means = train.mean(axis=axis)
    stds = train.std(axis=axis)
    train = (train - means) / stds
    test = (test - means) / stds
    return train, test


LICZBA_LITER_W_IMIENIU = 5
LICZBA_SPOLGLOSEK_W_IMIENIU = 3
TRAIN_SIZE = 0.9
EPOCH_COUNT = 20

data = pd.read_csv('tesla_2020_close.csv', header=None)
data = data.values

# Make datasets for t + 1 and t + k
X_t1, y_t1 = make_dataset(data, LICZBA_LITER_W_IMIENIU, 1)
X_tk, y_tk = make_dataset(data, LICZBA_LITER_W_IMIENIU, LICZBA_SPOLGLOSEK_W_IMIENIU)

# Split into train and test sets
t1_train_size = int(X_t1.shape[0] * TRAIN_SIZE)
X_train_t1, y_train_t1 = X_t1[:t1_train_size, ...], y_t1[:t1_train_size, ...]
X_test_t1, y_test_t1 = X_t1[t1_train_size:, ...], y_t1[t1_train_size:, ...]

tk_train_size = int(X_tk.shape[0] * TRAIN_SIZE)
X_train_tk, y_train_tk = X_tk[:tk_train_size, ...], y_tk[:tk_train_size, ...]
X_test_tk, y_test_tk = X_tk[tk_train_size:, ...], y_tk[tk_train_size:, ...]

# Normalize
X_train_t1, X_test_t1 = normalize_array(X_train_t1, X_test_t1, axis=0)
y_train_t1, y_test_t1 = normalize_array(y_train_t1, y_test_t1, axis=0)
X_train_tk, X_test_tk = normalize_array(X_train_tk, X_test_tk, axis=0)
y_train_tk, y_test_tk = normalize_array(y_train_tk, y_test_tk, axis=0)


def create_model_lstm(input_shape):
    input_tensor = output_tensor = Input(input_shape)
    output_tensor = TimeDistributed(Dense(32, activation='selu'))(output_tensor)
    output_tensor = TimeDistributed(Dense(32, activation='selu'))(output_tensor)
    output_tensor = LSTM(1, activation='selu', return_sequences=False)(output_tensor)
    model_t1 = Model(inputs=input_tensor, outputs=output_tensor)
    model_t1.compile(optimizer=RMSProp(), loss=MeanAbsolutePercentageError())
    return model_t1


model_lstm_t1 = create_model_lstm(X_train_t1.shape[1:])
model_lstm_tk = create_model_lstm(X_train_tk.shape[1:])


def create_model_gru(input_shape):
    input_tensor = output_tensor = Input(input_shape)
    output_tensor = TimeDistributed(Dense(32, activation='selu'))(output_tensor)
    output_tensor = GRU(1, activation='selu', return_sequences=False)(output_tensor)
    model_tk = Model(inputs=input_tensor, outputs=output_tensor)
    model_tk.compile(optimizer=RMSProp(), loss=MeanAbsolutePercentageError())
    return model_tk


model_gru_t1 = create_model_gru(X_train_t1.shape[1:])
model_gru_tk = create_model_gru(X_train_tk.shape[1:])

# Plot models
plot_model(model_lstm_t1, show_shapes=True, to_file='model_lstm_t1.png')
plot_model(model_lstm_tk, show_shapes=True, to_file='model_lstm_tk.png')
plot_model(model_gru_t1, show_shapes=True, to_file='model_gru_t1.png')
plot_model(model_gru_tk, show_shapes=True, to_file='model_gru_tk.png')

# Train the models
history_lstm_t1 = model_lstm_t1.fit(x=X_train_t1, y=y_train_t1, batch_size=32, epochs=EPOCH_COUNT,
                                    validation_data=(X_test_t1, y_test_t1))
history_lstm_tk = model_lstm_tk.fit(x=X_train_tk, y=y_train_tk, batch_size=32, epochs=EPOCH_COUNT,
                                    validation_data=(X_test_tk, y_test_tk))

history_gru_t1 = model_gru_t1.fit(x=X_train_t1, y=y_train_t1, batch_size=32, epochs=EPOCH_COUNT,
                                  validation_data=(X_test_t1, y_test_t1))
history_gru_tk = model_gru_tk.fit(x=X_train_tk, y=y_train_tk, batch_size=32, epochs=EPOCH_COUNT,
                                  validation_data=(X_test_tk, y_test_tk))

# Prediction
y_pred_model_lstm_t1 = model_lstm_t1.predict(X_test_t1)
y_pred_model_lstm_tk = model_lstm_tk.predict(X_test_tk)

y_pred_model_gru_t1 = model_gru_t1.predict(X_test_t1)
y_pred_model_gru_tk = model_gru_tk.predict(X_test_tk)


# Wizualizacja wyników uczenia
def plot_history(history, title):
    plt.title(title)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()


plot_history(history_lstm_t1, 'Model LSTM t+1')
plot_history(history_lstm_tk, 'Model LSTM t+k')
plot_history(history_gru_t1, 'Model GRU t+1')
plot_history(history_gru_tk, 'Model GRU t+k')


# Wizualizacja wyników pracy nauczonych modeli
def plot_predictions(y_pred, y_test, title):
    x = np.arange(y_pred.shape[0])
    plt.title(title)
    plt.plot(x, y_pred, label='Predykcje')
    plt.plot(x, y_test, label='Wartości prawdziwe')
    plt.legend()
    plt.show()


plot_predictions(y_pred_model_lstm_t1, y_test_t1, 'Predykcja t+1 dla modelu LSTM')
plot_predictions(y_pred_model_lstm_tk, y_test_tk, 'Predykcja t+k dla modelu LSTM')

plot_predictions(y_pred_model_gru_t1, y_test_t1, 'Predykcja t+1 dla modelu GRU')
plot_predictions(y_pred_model_gru_tk, y_test_tk, 'Predykcja t+k dla modelu GRU')

# Obliczenie błędu średniokwadratowego
ROUND_TO = 3
print(f'MAPE model LSTM t+1: {round(mean_absolute_percentage_error(y_test_t1, y_pred_model_lstm_t1) * 100, ROUND_TO)}%')
print(f'MAPE model LSTM t+k: {round(mean_absolute_percentage_error(y_test_tk, y_pred_model_lstm_tk) * 100, ROUND_TO)}%')
print(f'MAPE model GRU t+1: {round(mean_absolute_percentage_error(y_test_t1, y_pred_model_gru_t1) * 100, ROUND_TO)}%')
print(f'MAPE model GRU t+k: {round(mean_absolute_percentage_error(y_test_tk, y_pred_model_gru_tk) * 100, ROUND_TO)}%')


# Last loss value of each model
def get_last_loss(history):
    return history.history['loss'][-1]


def get_last_val_loss(history):
    return history.history['val_loss'][-1]


print(f'Last loss LSTM t+1: {round(get_last_loss(history_lstm_t1), ROUND_TO)}%')
print(f'Last loss LSTM t+k: {round(get_last_loss(history_lstm_tk), ROUND_TO)}%')
print(f'Last loss GRU t+1: {round(get_last_loss(history_gru_t1), ROUND_TO)}%')
print(f'Last loss GRU t+k: {round(get_last_loss(history_gru_tk), ROUND_TO)}%')

print(f'Last val loss LSTM t+1: {round(get_last_val_loss(history_lstm_t1), ROUND_TO)}%')
print(f'Last val loss LSTM t+k: {round(get_last_val_loss(history_lstm_tk), ROUND_TO)}%')
print(f'Last val loss GRU t+1: {round(get_last_val_loss(history_gru_t1), ROUND_TO)}%')
print(f'Last val loss GRU t+k: {round(get_last_val_loss(history_gru_tk), ROUND_TO)}%')

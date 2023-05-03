import numpy as np
import pandas as pd
from keras import Input, Model
from keras.layers import LSTM, TimeDistributed, Dropout, Dense, GRU
from keras.optimizers.optimizer_v2.rmsprop import RMSProp
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv('data/prices.csv')
descr = pd.read_csv('data/securities.csv')

data['date'] = pd.to_datetime(data['date'])
min_date = data['date'].min()
oldest_firms = data[data['date'] == min_date]['symbol']

selected_data = data[data['symbol'] == 'YHOO']
selected_data = selected_data.sort_values('date')
selected_data = selected_data.drop(columns=['date', 'symbol'])
# Columns: open, close, low, high, volume
selected_data = selected_data.values
print(selected_data.shape)


def make_dataset(dataset, obs_length, target_feature):
    total_ds_length = len(dataset)
    x_length = total_ds_length - obs_length
    x_ = np.zeros((x_length, obs_length, dataset.shape[1]))
    for i in range(x_length):
        x_[i, ...] = dataset[i:i + obs_length, ...]

    y_ = dataset[obs_length:, target_feature]
    return x_, y_


X, y = make_dataset(selected_data, 100, -3)
print(X.shape, y.shape)

train_size = int(X.shape[0] * 0.8)
X_train, y_train = X[:train_size, ...], y[:train_size, ...]
X_test, y_test = X[train_size:, ...], y[train_size:, ...]


def normalize_array(train, test, axis):
    means = train.mean(axis=axis)
    stds = train.mean(axis=axis)
    train = (train - means) / stds
    test = (test - means) / stds
    return train, test


X_train, X_test = normalize_array(X_train, X_test, 0)
y_train, y_test = normalize_array(y_train, y_test, 0)


def create_and_train_model(cell_type):
    # Tworzenie sieci LSTM lub GRU
    input_tensor = output_tensor = Input(X_train.shape[1:])
    output_tensor = TimeDistributed(Dense(32, activation='selu'))(output_tensor)
    output_tensor = TimeDistributed(Dense(32, activation='selu'))(output_tensor)
    output_tensor = TimeDistributed(Dropout(0.2))(output_tensor)

    if cell_type == 'LSTM':
        output_tensor = LSTM(1, activation='selu', return_sequences=False)(output_tensor)
    elif cell_type == 'GRU':
        output_tensor = GRU(1, activation='selu', return_sequences=False)(output_tensor)
    else:
        raise ValueError('Invalid cell type. Please specify either LSTM or GRU.')

    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(optimizer=RMSProp(), loss='MeanSquaredError', metrics=['MeanAbsoluteError'])
    print(model.summary())
    model.fit(x=X_train, y=y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))
    return model


lstm_model = create_and_train_model('LSTM')
gru_model = create_and_train_model('GRU')

# Wizualizacja wyników pracy nauczonych modeli
y_pred_lstm = lstm_model.predict(X_test).squeeze()
y_pred_gru = gru_model.predict(X_test).squeeze()
x = np.arange(y_pred_lstm.shape[0])
plt.title('Porównanie predykcji cen akcji za pomocą sieci LSTM oraz GRU')
plt.plot(x, y_pred_lstm, label='Predykcje LSTM')
plt.plot(x, y_pred_gru, label='Predykcje GRU')
plt.plot(x, y_test, label='Wartości prawdziwe')
plt.legend()
plt.show()

ROUND_TO = 4

# Wizualizacja różnic pomiędzy predykcjami a wartościami prawdziwymi
diff_pred_lstm = np.diff(y_pred_lstm.squeeze())
diff_pred_gru = np.diff(y_pred_gru.squeeze())
diff_true = np.diff(y_test)
diff_pred_lstm = np.sign(diff_pred_lstm)
diff_pred_gru = np.sign(diff_pred_gru)
diff_true = np.sign(diff_true)

print('Confusion matrix LSTM:')
print(confusion_matrix(diff_true, diff_pred_lstm))
print('Confusion matrix GRU:')
print(confusion_matrix(diff_true, diff_pred_gru))

print('Accuracy LSTM: ', round(accuracy_score(diff_true, diff_pred_lstm), ROUND_TO))
print('Accuracy GRU: ', round(accuracy_score(diff_true, diff_pred_gru), ROUND_TO))

# Analiza dokładności metodami: MSE, MAE, correlation coefficient, MAPE, R^2
print('MSE LSTM: ', round(mean_squared_error(y_test, y_pred_lstm), ROUND_TO))
print('MSE GRU: ', round(mean_squared_error(y_test, y_pred_gru), ROUND_TO))

print('MAE LSTM: ', round(mean_absolute_error(y_test, y_pred_lstm), ROUND_TO))
print('MAE GRU: ', round(mean_absolute_error(y_test, y_pred_gru), ROUND_TO))

print('Correlation coefficient LSTM: ', round(np.corrcoef(y_test, y_pred_lstm)[0, 1], ROUND_TO))
print('Correlation coefficient GRU: ', round(np.corrcoef(y_test, y_pred_gru)[0, 1], ROUND_TO))

print(f'MAPE LSTM: {round(mean_absolute_percentage_error(y_test, y_pred_lstm) * 100, ROUND_TO)}%')
print(f'MAPE GRU: {round(mean_absolute_percentage_error(y_test, y_pred_gru) * 100, ROUND_TO)}%')

print('R^2 LSTM: ', round(r2_score(y_test, y_pred_lstm), ROUND_TO))
print('R^2 GRU: ', round(r2_score(y_test, y_pred_gru), ROUND_TO))

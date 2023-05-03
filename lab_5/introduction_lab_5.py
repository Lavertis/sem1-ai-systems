import numpy as np
import pandas as pd
from keras import Input, Model
from keras.layers import GRU
from keras.optimizers.optimizer_v2.rmsprop import RMSProp
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

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

# Tworzenie sieci GRU
input_tensor = Input(X_train.shape[1:])
output_tensor = GRU(1, activation='selu', return_sequences=False)(input_tensor)
model = Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer=RMSProp(), loss='MeanSquaredError', metrics=['MeanAbsoluteError'])
print(model.summary())
model.fit(x=X_train, y=y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

# Wizualizacja wyników pracy nauczonego modelu GRU
y_pred = model.predict(X_test)
x = np.arange(y_pred.shape[0])
plt.title('Predykcja cen akcji za pomocą sieci GRU')
plt.plot(x, y_pred, label='Predykcje')
plt.plot(x, y_test, label='Wartości prawdziwe')
plt.legend()
plt.show()

# Wizualizacja predykcji spadków oraz wzrostów cen akcji
diff_pred = np.diff(y_pred.squeeze())
diff_true = np.diff(y_test)
diff_pred = np.sign(diff_pred)
diff_true = np.sign(diff_true)
print(confusion_matrix(diff_true, diff_pred))
print(accuracy_score(diff_true, diff_pred))

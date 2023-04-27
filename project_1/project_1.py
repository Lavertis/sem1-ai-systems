import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)


def prepare_data(file_path, look_back=1, validation_split=0.2):
    df = pd.read_csv(file_path, header=None, names=['Price'])
    dataset = np.array(df)

    # Utworzenie zestawów danych wejściowych i wyjściowych
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])

    # Przekształcenie danych wejściowych i wyjściowych na tablicę numpy
    x = np.array(data_x)
    y = np.array(data_y)

    # Podział danych na zbiór treningowy i testowy
    train_size = int(len(x) * (1 - validation_split))
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return x_train, y_train, x_test, y_test


X_train, Y_train, X_test, Y_test = prepare_data('data.csv', look_back=1, validation_split=0.2)
input_dim = X_train.shape[1]
output_dim = 1


def create_model(hidden_layers=1, learning_rate=0.0001):
    model = Sequential()
    model.add(Dense(1, input_dim=input_dim, activation='linear', kernel_initializer='ones', bias_initializer='zeros'))
    for _ in range(hidden_layers):
        model.add(Dense(1, activation='linear', kernel_initializer='ones', bias_initializer='zeros'))
    model.add(Dense(output_dim, activation='linear', kernel_initializer='ones', bias_initializer='zeros'))
    model.compile(loss='mape', optimizer=Adam(learning_rate), metrics=['mape'])
    return model


def show_prediction_plot_with_true_values(y_test, y_pred):
    plt.plot(y_test)
    plt.plot(y_pred)
    plt.title('Predykcja')
    plt.ylabel('Cena')
    plt.xlabel('Dzień')
    plt.legend(['Dane testowe', 'Predykcja'], loc='upper left')
    plt.show()


def plot_mean_absolute_percentage_error(history):
    plt.plot(history.history['mape'])
    plt.plot(history.history['val_mape'])
    plt.title('MAPE modelu')
    plt.ylabel('MAPE')
    plt.xlabel('Epoka')
    plt.legend(['Zbiór treningowy', 'Zbiór walidacyjny'], loc='upper left')
    plt.show()


histories = []
fitted_models = []
params_grid = {
    'hidden_layers': [4, 8, 16],
    'learning_rate': [0.000001, 0.00001, 0.0001]
}
models = [
    create_model(hidden_layers=hl, learning_rate=lr)
    for hl in params_grid['hidden_layers']
    for lr in params_grid['learning_rate']
]

mean_mapes = []
for model in models:
    weights = model.get_weights()
    model_mapes = []
    model_histories = []
    for train, test in KFold(n_splits=5).split(X_train, Y_train):
        model.set_weights(weights)
        history = model.fit(X_train[train], Y_train[train], epochs=5, batch_size=32, validation_split=0.3, verbose=1)

        model_mapes.append(history.history['val_mape'][-1])
        model_histories.append(history)

    model_mapes_avg = sum(model_mapes) / len(model_mapes)
    mean_mapes.append(round(model_mapes_avg, 4))
    histories.append(model_histories)
    fitted_models.append(model)

print("Mean MAPE for each model:")
for i, mape in enumerate(mean_mapes):
    print(f"Model {i} MAPE: {mape}")
    print(f"Model {i} params: "
          f"Hidden layers - {params_grid['hidden_layers'][i // len(params_grid['learning_rate'])]}, "
          f"Learning rate - {params_grid['learning_rate'][i % len(params_grid['learning_rate'])]}")

best_model_index = np.argmin(mean_mapes)
best_model = fitted_models[best_model_index]
best_model_histories = histories[best_model_index]

for i, history in enumerate(best_model_histories):
    plt.title(f'Training and validation loss for cv fold {i}')
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()

best_history = best_model_histories[np.argmin([history.history['val_mape'][-1] for history in best_model_histories])]
plot_mean_absolute_percentage_error(best_history)
show_prediction_plot_with_true_values(Y_test, best_model.predict(X_test))
best_model.save('model.h5')

print(f"Best model MAPE: {min(mean_mapes)}")
print(
    f"Best model params: "
    f"Hidden layers - {params_grid['hidden_layers'][best_model_index // len(params_grid['learning_rate'])]}, "
    f"Learning rate -  {params_grid['learning_rate'][best_model_index % len(params_grid['learning_rate'])]}"
)

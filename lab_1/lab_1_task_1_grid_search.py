import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam, SGD
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Import danych
dataset = pd.read_csv('data/train.csv')
dataset.head(10)

# 2. Zamiana pandas.DataFrame na numpy.array
X = dataset.iloc[:, :20].values
y = dataset.iloc[:, 20:21].values

# 3. Normalizacja danych treningowych
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. Zakodowanie klas wynikowych
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

# 5. Podział danych na zbiór treningowy i walidacyjny (90% treningowy, 10% walidacyjny)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# 6. Stwórz sieć neuronową składającą się z 3 warstw (16/12/4 neuronów)
class_num = y.shape[1]


def create_model(num_layers=1, num_neurons=8, activation='relu', optimizer=Adam, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(num_neurons, input_shape=(X.shape[1],), activation=activation))
    for _ in range(num_layers - 1):
        model.add(Dense(num_neurons, activation=activation))
    model.add(Dense(class_num, activation='softmax'))
    model.compile(optimizer=optimizer(learning_rate), loss=CategoricalCrossentropy(), metrics=['accuracy'])
    return model


keras_model = KerasClassifier(
    model=create_model,
    verbose=1,
    activation='relu',
    optimizer=Adam,
    learning_rate=0.001,
    num_layers=1,
    num_neurons=8,
    loss=CategoricalCrossentropy(),
    metrics=['accuracy'],
    epochs=1
)

# Define the hyperparameters to test
param_grid = {
    'num_layers': [2, 3],
    'num_neurons': [16, 32],
    'activation': ['sigmoid', 'relu'],
    'optimizer': [Adam, SGD],
    'learning_rate': [0.001, 0.01],
}

# Perform the grid search
grid = GridSearchCV(estimator=keras_model, param_grid=param_grid, cv=5)
grid_result = grid.fit(X_train, y_train)

# Print the best hyperparameters and corresponding accuracy score
print(f"Best parameters: {grid_result.best_params_}")
print(f"Best accuracy: {grid_result.best_score_}")

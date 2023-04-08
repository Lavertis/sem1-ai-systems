import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
class_num = y.shape[1]  # 4
model = Sequential()
model.add(Dense(16, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(class_num, activation='softmax'))

# 7. Określenie wymaganych elementów kompilacji modelu [loss, optimizer, metrics]
learning_rate = 0.0001
model.compile(optimizer=Adam(learning_rate), loss=CategoricalCrossentropy(), metrics=['accuracy'])

# 8. Trenowanie modelu
model.fit(X_train, y_train, batch_size=15, epochs=100, validation_data=(X_test, y_test), verbose=1)

# 9. Testowanie modelu na zbiorze testowym
y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)
y_test = y_test.argmax(axis=1)
print(accuracy_score(y_test, y_pred))

# 10. Wykonaj walidację krzyżową - napisz skrypt, który pozwoli na dokonanie wyszukiwania
# siatkowego w celu znalezienia najbardziej obiecujących wartości hiperparametrów.
# Przetestuj takie parametry, jako liczba warstw, liczba neuronów w warstwie, funkcja
# aktywacji, optymalizator, prędkość nauczania. Uwzględnij zjawisko przetrenowania –
# oznacza to, że najlepszy wynik nie koniecznie będzie po ostatniej epoce uczenia sieci.

accs = []
weights = model.get_weights()
for train_index, val_index in KFold(5).split(X_train):
    X_train_cv = X_train[train_index, :]
    X_val_cv = X_train[val_index, :]
    y_train_cv = y_train[train_index, :]
    y_val_cv = y_train[val_index, :]

    model.set_weights(weights)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model.fit(X_train_cv, y_train_cv, batch_size=15, epochs=15,
              validation_data=(X_val_cv, y_val_cv), callbacks=[early_stopping], verbose=1)

    y_pred_cv = model.predict(X_val_cv).argmax(axis=1)
    y_val_cv = y_val_cv.argmax(axis=1)
    accs.append(accuracy_score(y_val_cv, y_pred_cv))

print(accs)
print(np.mean(accs))

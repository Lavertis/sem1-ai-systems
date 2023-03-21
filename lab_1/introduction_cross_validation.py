import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

data = load_iris()
y = data.target
x = data.data
y = pd.Categorical(y)
y = pd.get_dummies(y).values
class_num = y.shape[1]

model = Sequential()
model.add(Dense(64, input_shape=(x.shape[1],), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(class_num, activation='softmax'))
learning_rate = 0.0001
model.compile(optimizer=Adam(learning_rate), loss=CategoricalCrossentropy(), metrics=['accuracy'])

scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
weights = model.get_weights()
accs = []
for train_index, val_index in KFold(5).split(x_train):
    x_train_cv = x_train[train_index, :]
    x_val_cv = x_train[val_index, :]
    y_train_cv = y_train[train_index, :]
    y_val_cv = y_train[val_index, :]

    x_train_cv = scaler.fit_transform(x_train_cv)
    x_val_cv = scaler.transform(x_val_cv)

    model.set_weights(weights)
    model.fit(x_train_cv, y_train_cv, batch_size=15, epochs=100, validation_data=(x_val_cv, y_val_cv), verbose=1)

    y_pred_cv = model.predict(x_val_cv).argmax(axis=1)
    y_val_cv = y_val_cv.argmax(axis=1)
    accs.append(accuracy_score(y_val_cv, y_pred_cv))

print(accs)
print(np.mean(accs))

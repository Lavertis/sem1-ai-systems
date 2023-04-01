import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras.utils import plot_model
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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

model.summary()
plot_model(model, to_file="my_model.png")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
history = model.fit(x_train, y_train, batch_size=15, epochs=100, validation_data=(x_test, y_test), verbose=2)

historia = model.history.history
floss_train = historia['loss']
floss_test = historia['val_loss']
acc_train = historia['accuracy']
acc_test = historia['val_accuracy']
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
epochs = np.arange(0, 100)
ax[0].plot(epochs, floss_train, label='floss_train')
ax[0].plot(epochs, floss_test, label='floss_test')
ax[0].set_title('Funkcje strat')
ax[0].legend()
ax[1].set_title('Dokladnosci')
ax[1].plot(epochs, acc_train, label='acc_train')
ax[1].plot(epochs, acc_test, label='acc_test')
ax[1].legend()
plt.show()

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
print(y_pred)
print(y_test)
print(y_pred == y_test)
print(accuracy_score(y_test, y_pred))

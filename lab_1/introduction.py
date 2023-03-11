import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

data = load_iris()
y = data.target
X = data.data
y = pd.Categorical(y)
y = pd.get_dummies(y).values
class_num = y.shape[1]

model = Sequential()
model.add(Dense(64, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(class_num, activation='softmax'))

learning_rate = 0.0001
model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics='accuracy')
model.summary()
plot_model(model, to_file="my_model.png")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
history = model.fit(X_train, y_train, batch_size=15, epochs=100, validation_data=(X_test, y_test), verbose=2)

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

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
print(y_pred)
print(y_test)
print(y_pred == y_test)
print()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# accs = []
# scaler = StandardScaler()
# for train_index, test_index in KFold(5).split(X_train):
#     X_train_cv = X_train[train_index, :]
#     X_test_cv = X_train[test_index, :]
#     y_train_cv = y_train[train_index, :]
#     y_test_cv = y_train[test_index, :]
#     X_train_cv = scaler.fit_transform(X_train_cv)
#     X_test_cv = scaler.transform(X_test_cv)
#     model.fit(X_train, y_train, batch_size=15, epochs=100, validation_data=(X_test, y_test), verbose=2)

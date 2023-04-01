import numpy as np
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

WITH_POOLING = True

train, test = mnist.load_data()
train = (train[0][:3000], train[1][:3000])
test = (test[0][:1000], test[1][:1000])

x_train, y_train = train[0], train[1]
x_test, y_test = test[0], test[1]
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

epochs = 10
class_cnt = np.unique(y_train).shape[0]  # 10, because we have 10 digits
filter_cnt = 32
learning_rate = 0.0001
act_func = 'relu'
kernel_size = (3, 3)
model = Sequential()
conv_rule = 'same'
model.add(Conv2D(
    input_shape=x_train.shape[1:],
    filters=filter_cnt,
    kernel_size=kernel_size,
    padding=conv_rule,
    activation=act_func
))
if WITH_POOLING:
    pooling_size = (2, 2)
    model.add(MaxPooling2D(pooling_size))  # max is better than average, because borders are more important
model.add(Flatten())
model.add(Dense(class_cnt, activation='softmax'))
model.compile(optimizer=Adam(learning_rate), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()
model.fit(x=x_train, y=y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=1)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
print('Accuracy: ', np.sum(y_pred == y_test) / y_test.shape[0])

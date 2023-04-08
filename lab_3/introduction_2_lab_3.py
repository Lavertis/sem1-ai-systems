import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input
from keras.losses import CategoricalCrossentropy
from keras.models import Model
from keras.utils import plot_model

# Sequential model using the functional API with layers in a list

data = mnist.load_data()
x_train, y_train = data[0][0], data[0][1]
x_test, y_test = data[1][0], data[1][1]
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
y_train = pd.get_dummies(pd.Categorical(y_train)).values
y_test = pd.get_dummies(pd.Categorical(y_test)).values

filter_cnt = 32
kernel_size = (3, 3)
act_func = 'selu'
class_cnt = y_train.shape[1]

layers = [
    Conv2D(filter_cnt, kernel_size, activation=act_func),
    MaxPooling2D(2, 2),
    Conv2D(filter_cnt, kernel_size, activation=act_func),
    MaxPooling2D(2, 2),
    Conv2D(filter_cnt, kernel_size, activation=act_func),
    GlobalAveragePooling2D(),
    Dense(class_cnt, activation='softmax')
]
output_tensor = input_tensor = Input(x_train.shape[1:])
for layer in layers:
    output_tensor = layer(output_tensor)

ANN = Model(inputs=input_tensor, outputs=output_tensor)
ANN.compile(loss=CategoricalCrossentropy(), metrics='accuracy', optimizer='adam')
plot_model(ANN, show_shapes=True)

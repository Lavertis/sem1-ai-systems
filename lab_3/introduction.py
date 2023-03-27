import numpy as np
import pandas as pd
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input, concatenate
from keras.layers import Lambda
from keras.models import Model
from keras.utils import plot_model


def add_inception_module(input_tensor_):
    act_func_ = 'relu'
    paths = [
        [
            Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation=act_func_)
        ],
        [
            Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation=act_func_),
            Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation=act_func_)
        ],
        [
            Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation=act_func_),
            Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation=act_func_)
        ],
        [
            MaxPooling2D(pool_size=(3, 3), strides=1, padding='same'),
            Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation=act_func_)
        ]
    ]
    for_concat = []

    for path in paths:
        output_tensor_ = input_tensor_
        for layer_ in path:
            output_tensor_ = layer_(output_tensor_)
            for_concat.append(output_tensor_)

    return concatenate(for_concat)


def ReLOGU(tensor):
    mask = tensor >= 1
    tensor = tf.where(mask, tensor, 1)
    tensor = tf.math.log(tensor)
    return tensor


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
output_tensor = input_tensor = Input(x_train.shape[1:])
output_tensor = Conv2D(filter_cnt, kernel_size, activation=act_func)(output_tensor)
output_tensor = MaxPooling2D(2, 2)(output_tensor)
output_tensor = Conv2D(filter_cnt, kernel_size, activation=act_func)(output_tensor)
output_tensor = MaxPooling2D(2, 2)(output_tensor)
output_tensor = Conv2D(filter_cnt, kernel_size, activation=act_func)(output_tensor)
output_tensor = GlobalAveragePooling2D()(output_tensor)
output_tensor = Dense(class_cnt, activation='softmax')(output_tensor)
ANN = Model(inputs=input_tensor, outputs=output_tensor)
ANN.compile(loss='categorical_crossentropy', metrics='accuracy', optimizer='adam')

layers = [Conv2D(filter_cnt, kernel_size, activation=act_func),
          MaxPooling2D(2, 2),
          Conv2D(filter_cnt, kernel_size, activation=act_func),
          MaxPooling2D(2, 2),
          Conv2D(filter_cnt, kernel_size, activation=act_func),
          GlobalAveragePooling2D(),
          Dense(class_cnt, activation='softmax')]
output_tensor = input_tensor = Input(x_train.shape[1:])
for layer in layers:
    output_tensor = layer(output_tensor)

output_tensor = input_tensor = Input(x_train.shape[1:])
incept_module_cnt = 2
for i in range(incept_module_cnt):
    output_tensor = add_inception_module(output_tensor)
output_tensor = GlobalAveragePooling2D()(output_tensor)
output_tensor = Dense(class_cnt, activation='softmax')(output_tensor)
ANN = Model(inputs=input_tensor, outputs=output_tensor)
ANN.compile(loss='categorical_crossentropy', metrics='accuracy', optimizer='adam')

plot_model(ANN, show_shapes=True)

output_tensor = input_tensor = Input(x_train.shape[1:])
incept_module_cnt = 2
for i in range(incept_module_cnt):
    output_tensor = add_inception_module(output_tensor)
output_tensor = Conv2D(32, (3, 3))(output_tensor)
output_tensor = Lambda(ReLOGU)(output_tensor)
output_tensor = GlobalAveragePooling2D()(output_tensor)
output_tensor = Dense(class_cnt, activation='softmax')(output_tensor)
ANN = Model(inputs=input_tensor, outputs=output_tensor)
ANN.compile(loss='categorical_crossentropy', metrics='accuracy', optimizer='adam')

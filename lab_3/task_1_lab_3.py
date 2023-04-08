from keras import Input, Model
from keras.layers import Reshape, BatchNormalization, Dense, Average
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras.utils import plot_model

output_tensor = input_tensor = Input((28, 28))
output_tensor = Reshape([784])(output_tensor)
output_tensor = BatchNormalization()(output_tensor)

paths = [
    [
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(16, activation='relu'),
        Dense(10, activation='softmax')
    ],
    [
        Dense(512, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ],
    [
        Dense(512, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ],
    [
        Dense(512, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ],
    [
        Dense(512, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ]
]

path_output_tensors = []
for path in paths:
    temp_tensor = output_tensor
    for layer in path:
        temp_tensor = layer(temp_tensor)
    path_output_tensors.append(temp_tensor)

output_tensor = Average()(path_output_tensors)
ANN = Model(inputs=input_tensor, outputs=output_tensor)
ANN.compile(loss=CategoricalCrossentropy(), metrics='accuracy', optimizer=Adam())
plot_model(ANN, show_shapes=True)

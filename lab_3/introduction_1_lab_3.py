import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input
from keras.losses import CategoricalCrossentropy
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.metrics import f1_score, confusion_matrix

# Sequential model using the functional API

data = mnist.load_data()
x_train, y_train = data[0][0], data[0][1]
x_test, y_test = data[1][0], data[1][1]

# Reduce the size of the dataset
x_train, y_train = x_train[:10000], y_train[:10000]
x_test, y_test = x_test[:1000], y_test[:1000]

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
ANN.compile(loss=CategoricalCrossentropy(), metrics='accuracy', optimizer=Adam())
plot_model(ANN, show_shapes=True)

ANN.fit(x_train, y_train, epochs=3, batch_size=128, validation_data=(x_test, y_test))
y_pred = ANN.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
score_f1 = f1_score(y_test, y_pred, average='macro')
score_accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
conf_matrix = confusion_matrix(y_test, y_pred)
print('F1 score: ', score_f1)
print('Accuracy: ', score_accuracy)
print('Confusion matrix:')
print(conf_matrix)

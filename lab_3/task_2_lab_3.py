from keras.applications.resnet import ResNet50
from keras.datasets import cifar10
from keras.layers import Dense, Flatten
from keras.losses import CategoricalCrossentropy
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

# Load the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Reduce the data size
train_size = 5_000
test_size = 1_000
x_train, y_train = x_train[:train_size], y_train[:train_size]
x_test, y_test = x_test[:test_size], y_test[:test_size]

# Normalize the data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Convert the labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the ResNet model
resnet = ResNet50(include_top=False, input_shape=(32, 32, 3))
x = Flatten()(resnet.output)
x = Dense(10, activation='softmax')(x)
model = Model(resnet.input, x)

# Compile the model
model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

import numpy as np
from keras import Sequential, layers
from keras.datasets import cifar10
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train, y_train = x_train[:3000], y_train[:3000]
# x_test, y_test = x_test[:500], y_test[:500]

img_height = img_width = 32

class_count = len(np.unique(y_train))
model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(class_count, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()

epochs = 5
batch_size = 32
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
score_f1 = f1_score(y_test, y_pred, average='macro')
score_accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print('F1 score: ', score_f1)
print('Accuracy score: ', score_accuracy)
print('Confusion matrix:')
print(conf_matrix)

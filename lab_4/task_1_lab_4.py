import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras import Input, Model
from keras.layers import Conv2D, Dropout, UpSampling2D
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.utils import plot_model
from matplotlib import pyplot as plt

# Load the PASCAL VOC 2012 dataset
train_data, info = tfds.load('voc/2012', split='train', with_info=True)


# Preprocessing function for the images and masks
def preprocess(item):
    image = tf.cast(item['image'], tf.float32) / 255.
    mask = item['segmentation_mask']
    # Map the mask labels to 0-20 range
    mask -= 1
    mask = tf.cast(mask, tf.int32)
    return image, mask


# Preprocess the dataset
train_data = train_data.map(preprocess).batch(16)

# Define the input shape
input_shape = (256, 256, 3)

# Define the number of classes
num_classes = 20

# Define the input tensor
input_tensor = Input(shape=input_shape)

# Define the encoder part of the model
encoder = tf.keras.applications.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

# Define the decoder part of the model
x = encoder.output
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Dropout(0.5)(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Dropout(0.5)(x)
x = Conv2D(num_classes, (1, 1), activation='linear', padding='valid')(x)
x = UpSampling2D(size=(32, 32), interpolation='bilinear')(x)

# Define the model
model = Model(inputs=encoder.input, outputs=x)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Plot the model architecture
plot_model(model, show_shapes=True, show_layer_names=True)

# Train the model
history = model.fit(train_data, epochs=10)

# Evaluate the model on a few examples
for example in train_data.take(3):
    image, mask = example
    predicted_mask = model.predict(image)
    predicted_mask = np.argmax(predicted_mask, axis=-1)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image[0])
    plt.title('Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(mask[0], cmap='nipy_spectral')
    plt.title('True Mask')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask[0], cmap='nipy_spectral')
    plt.title('Predicted Mask')
    plt.axis('off')
    plt.show()

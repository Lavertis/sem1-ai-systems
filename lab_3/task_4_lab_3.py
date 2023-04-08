import numpy as np
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, Flatten
from keras.losses import CategoricalCrossentropy
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Set the path to the data directory
train_dir = '../lab_2/data/flower_photos'
test_dir = '../lab_2/data/flower_photos'

# Set the image size and batch size
img_size = (224, 224)
batch_size = 16
validation_split = 0.2

# Create a data generator for the training data
train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=validation_split)
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Create a data generator for the test data
test_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=validation_split)
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

class_names = np.unique(train_generator.classes)
num_classes = len(class_names)

# Build the ResNet model
resnet = DenseNet121(include_top=False, input_shape=(img_size[0], img_size[1], 3))
x = Flatten()(resnet.output)
x = Dense(num_classes, activation='softmax')(x)
model = Model(resnet.input, x)

# Compile the model
model.compile(optimizer=Adam(0.0001), loss=CategoricalCrossentropy(), metrics=['accuracy'])

# Train the model
model.fit(
    x=train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=5,
    validation_data=test_generator,
    validation_steps=test_generator.n // test_generator.batch_size
)

# Evaluate the model
score = model.evaluate(test_generator, steps=test_generator.n // test_generator.batch_size, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

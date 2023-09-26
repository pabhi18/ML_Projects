import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import layers, models
from keras.optimizers import legacy
from PIL import Image
from tensorflow.keras.optimizers.legacy import Adam

from keras.models import Sequential 
from keras.layers import Convolution2D 
from keras.layers import MaxPooling2D 
from keras.layers import Flatten
from keras.layers import Dense, Dropout

train_datagen = ImageDataGenerator(rescale=1/255)
val_datagen = ImageDataGenerator(rescale=1/255)

image_size = (48, 48)
batch_size = 64

train_dir = 'Data/train'
val_dir = 'Data/test'

train_genrator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical',
)
validation_genrator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical',
)

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape = (48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.50),
    layers.Dense(7, activation='softmax'),
])

cnn.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
emotion_model_info = cnn.fit(train_genrator, steps_per_epoch=28709/64,
        epochs=50, validation_data=validation_genrator,  validation_steps=7178/64)

cnn.save('model.h5')










# -- This is yohans work space --

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os


train = ImageDataGenerator(rescale=1/255)
validation =  ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory(
    "yohan\\train", 
    target_size=(200,200), 
    batch_size = 3,   
    class_mode="binary"
)

validate_dataset = train.flow_from_directory(
    "yohan\\validate", 
    target_size=(200,200), 
    batch_size = 3, 
    class_mode="binary"
)

print(validate_dataset.class_indices)

# Create model
model = tf.keras.models.Sequential(
    [
        # Layer1
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(200,200,3)), 
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2), # MaxPool takes the max pixel out of given pixels

        # Layer2
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'), 
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        # Layer3
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'), 
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        # Flatten
        tf.keras.layers.Flatten(),

        # Apply 2 dense layers
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'), # sigmoid becuase program is binary (500 or 1000)
    ]
)

# Compile model
model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model_fit = model.fit(train_dataset, steps_per_epoch=3, epochs=8, validation_data=validate_dataset) # Epoch is the number of iterations to run compile model

# Test model classification 
dir_path = "yohan\\test"


img = image.load_img("yohan\\test\\Five Thousand Rupee Fake.jpg", target_size=(200,200))

X = image.img_to_array(img)
X = np.expand_dims(X, axis=0)
images = np.vstack([X])
val = model.predict(images)


## In validate_dataset.class_indices Fake = 0, and Real = 500
if val == 0:
    print(f"Fake")
else:
    print(f"Real")
        
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib as plt

# Define the path to the dataset
dataset_path = "C:\\Users\\yohan\\OneDrive\\Desktop\\split_dataset"

# Load the InceptionV3 model with pre-trained weights
base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(512, 512, 3))

# Freeze all the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add a few custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(6, activation="softmax")(x)

# Build the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with categorical crossentropy loss and Adam optimizer
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Define the data generators for training and testing data
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size=32

train_generator = train_datagen.flow_from_directory(
    dataset_path + "/train",
    target_size=(512, 512),
    batch_size=batch_size,
    class_mode="categorical")

test_generator = test_datagen.flow_from_directory(
    dataset_path + "/test",
    target_size=(512,512),
    batch_size=batch_size,
    class_mode="categorical")

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Add a few more callbacks to reduce overfitting
# Reduce the learning rate when the learning plateaus
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)

# Add early stopping to the callbacks list
callbacks = [early_stopping, reduce_lr]

# Train the model
history = model.fit(
    train_generator,
    epochs=75,
    steps_per_epoch=len(train_generator),
    validation_data=test_generator,
    validation_steps=len(test_generator),
    callbacks=callbacks)

# Plot the accuracy and loss plots for training and validation data
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy Plot")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss Plot")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

#model.save("C:\\Users\\yohan\\OneDrive\\Desktop\\models\\InceptionV3_512_32-new-v1.1.h5")
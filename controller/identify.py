from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load the saved models
model = load_model("../model/weights/InceptionV3_512_32-new-v1.1.h5")

# Define the path to the validation dataset
validation_path = "C:\\Users\\yohan\\OneDrive\\Desktop\\split_dataset\\validate"

# Define the data generator for validation data
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    validation_path,
    target_size=(512, 512),
    batch_size=32,
    class_mode=None,
    shuffle=False)

# Make predictions on the validation data
predictions = model.predict(validation_generator, steps=len(validation_generator), verbose=1)

# Convert the predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)

# Get the class indices
class_indices = validation_generator.class_indices

# Invert the class indices to get a mapping from class indices to class labels
class_labels = {v: k for k, v in class_indices.items()}

correct = 0

# Print the class labels for each image
for i, image_path in enumerate(validation_generator.filepaths):
    if (image_path.split("\\")[-1]).split(" ")[0] == class_labels[predicted_classes[i]]:
        correct +=1
    print(f"Image {i + 1}: {image_path}, Class Label: {class_labels[predicted_classes[i]]}")

print(f"Total correct: {correct}")
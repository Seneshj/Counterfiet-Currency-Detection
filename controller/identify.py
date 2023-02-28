from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pyttsx3
import cv2
import pandas as pd
import numpy as np

# Initialize the camera
print(">> Getting the camera ready >>")
cap = cv2.VideoCapture(0)

print(">> Loading model >>")
# Load the saved models
model = load_model("C:\\Users\\yohan\\OneDrive\\Desktop\\models\\latest_inceptionv3.h5")


def voice_out(phrase):
    # Initialize the engine
    engine = pyttsx3.init()

    engine.setProperty('rate', 150)  # Change the rate to adjust the speed of speech

    # Set the voice properties (optional)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  # Change the index to switch voices

    # Say the text
    engine.say(phrase)

    # Run the engine and wait for the speech to finish
    engine.runAndWait()


def capture_image():
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Display the captured frame
    cv2.imshow('frame', frame)

    # Check for user input to capture the image
    if cv2.waitKey(1) & 0xFF == ord('c'):
        # Save the captured image
        cv2.imwrite('image_to_predict.jpg', frame)
        # close the window
        cv2.destroyAllWindows()
        return True
    else:
        return False


def predict_value(model):
    # Create a Pandas DataFrame containing the path of the image to predict
    image_path = "image_to_predict.jpg"
    data = pd.DataFrame({'filename': [image_path]})

    predict_datagen = ImageDataGenerator(rescale=1. / 255)
    predict_generator = predict_datagen.flow_from_dataframe(
        data,
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=(512, 512),
        batch_size=1,
        shuffle=False,
        preprocessing_function=lambda x: x / 255.0 - 0.5
    )

    # Make predictions on the image to predict
    predictions = model.predict(predict_generator, steps=len(predict_generator), verbose=1)

    # Convert the predictions to class labels
    predicted_class_index = np.argmax(predictions)

    classes = ["100", "1000", "20", "50", "500", "5000"]
    value = classes[predicted_class_index]

    return value


value = 0


def main():
    global value
    # Start the prediction loop
    while True:
        # Capture an image from the camera
        if not capture_image():
            continue
        else:
            # Predict the value of the captured image
            value = predict_value(model)

            # Speak out the value
            voice_out(value)
            break


def get_value():
    return value

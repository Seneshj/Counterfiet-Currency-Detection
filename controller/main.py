from keras.models import load_model
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import pyttsx3

# Load the saved models
model = load_model("C:\\Users\\yohan\\OneDrive\\Desktop\\models\\latest_inceptionv3.h5")

# Define the data generator for the image to predict
predict_datagen = ImageDataGenerator(rescale=1. / 255)

# Initialize the camera
cap = cv2.VideoCapture(0)


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


# Start the prediction loop
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Display the captured frame
    cv2.imshow('frame', frame)

    # Check for user input to capture the image
    if cv2.waitKey(1) & 0xFF == ord('c'):
        # Save the captured image
        cv2.imwrite('image_to_predict.jpg', frame)

        # Create a Pandas DataFrame containing the path of the image to predict
        image_path = "image_to_predict.jpg"
        data = pd.DataFrame({'filename': [image_path]})

        predict_generator = predict_datagen.flow_from_dataframe(
            data,
            x_col='filename',
            y_col=None,
            class_mode=None,
            target_size=(512, 512),
            batch_size=1,
            shuffle=False
        )

        # Make predictions on the image to predict
        predictions = model.predict(predict_generator, steps=len(predict_generator), verbose=1)

        # Convert the predictions to class labels
        predicted_class_index = np.argmax(predictions)

        classes = ["100", "1000", "20", "50", "500", "5000"]
        value = classes[predicted_class_index]

        # Output the predicted value
        print(value)

        # Speak the predicted value
        voice_out("I have predicted that the note you scanned is a "+value+" rupee note.")

        # Prompt the user to continue or not
        while True:
            voice_out("Type y in the console if you wish to scan again")
            user_input = input("Do you want to continue? (y/n): ")
            if user_input.lower() == "n":
                # Release the camera and close the window
                cap.release()
                cv2.destroyAllWindows()
                exit()
            elif user_input.lower() == "y":
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")


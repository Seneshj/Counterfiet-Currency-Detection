import cv2
import pyttsx3

# Initialize the camera
print(">> Getting the camera ready >>")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def capture_image(name):
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Display the captured frame
    cv2.imshow('frame', frame)

    # Check for user input to capture the image
    if cv2.waitKey(1) & 0xFF == ord('c'):
        # Save the captured image
        cv2.imwrite(f'{name}.jpg', frame)
        cv2.destroyAllWindows()
        return True
    else:
        return False


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

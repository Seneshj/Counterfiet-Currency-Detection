import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import arduino


# note value is the value of the note as an int or string. Eg: 20,50,100,500,1000,5000
def fake_classifier(note_value):
    arduino.uv_on()
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))
    # assuming the models are saved in the same file as the python file
    model = keras.models.load_model('../model/weights/fake-' + str(note_value) + '.h5')
    yhat = model.predict(np.expand_dims(resized / 255, 0))
    arduino.uv_off()
    # returns a string saying real or fake
    if yhat[0] > 0.5:
        return 'Real'
    else:
        return 'Fake'

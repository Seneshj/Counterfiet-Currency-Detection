import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

def detect_note():
    model = keras.models.load_model('crop.h5')
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _,frame = cap.read()
        frame = frame[50:500,50:500,:]
        resized = tf.image.resize(frame,(120,120))
        yhat = model.predict(np.expand_dims(resized / 255,0))
        if yhat[0] > 0.5:
            return True
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


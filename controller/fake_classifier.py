import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2


# note value is the value of the note as an int or string. Eg: 20,50,100,500,1000,5000
# uv_image is the loaded image from the image file. Eg: uv_image = cv2.imread('...')
def fake_classifier(note_value, uv_image):
    rgb = cv2.cvtColor(uv_image, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))
    # assuming the models are saved in the same file as the python file
    model = keras.models.load_model('fake-' + str(note_value) + '.h5')
    yhat = model.predict(np.expand_dims(resized / 255, 0))
    # returns a string saying real or fake
    if yhat[0] > 0.5:
        return 'Real'
    else:
        return 'Fake'

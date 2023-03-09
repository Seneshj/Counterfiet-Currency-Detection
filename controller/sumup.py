from tensorflow.keras.models import load_model
import detect_note
import identify

# Load the pre-trained note recognition model
model = load_model('model_weights.h5')

notes =[] # List to store the detected notes
sum_of_notes = 0.0 # Variable to keep track the sum of the detected notes





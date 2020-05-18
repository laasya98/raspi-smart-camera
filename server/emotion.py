from keras.models import load_model
import numpy as np

def detect_emotion(face_image):
    model = load_model("./emotion_detector_models/model_v6_23.hdf5")
    predicted_class = np.argmax(model.predict(face_image))
    print ("predicted class" + str(predicted_class))
    return predicted_class
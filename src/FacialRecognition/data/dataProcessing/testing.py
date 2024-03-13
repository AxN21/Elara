# Imports
import numpy as np
import tensorflow as tf
from model import L1Dist
from dataProcessing import preprocess
import os
import cv2

# Define constants
APP_DATA = '/home/axn/Desktop/Python projects/Elara/src/FacialRecognition/application_data'
VER_DATA = '/home/axn/Desktop/Python projects/Elara/src/FacialRecognition/application_data/verification_images'
INP_IMG = '/home/axn/Desktop/Python projects/Elara/src/FacialRecognition/application_data/input_images'
SAVE_MODEL_PATH = '/home/axn/Desktop/Python projects/Elara'

# Load model | Loss function | Optimizations
model = tf.keras.models.load_model(SAVE_MODEL_PATH, custom_objects={'L1Dist':L1Dist})
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)
model.compile(optimizer=opt, loss=binary_cross_loss, metrics=['accuracy'])


# Verification function
def verify(model, detection_threshold, verification_threshold):
    #build results array
    results = []
    for image in os.listdir(os.path.join(APP_DATA, VER_DATA)):
        input_img = preprocess(os.path.join(APP_DATA, INP_IMG, 'input_image.jpg'))
        validation_img = preprocess(os.path.join(APP_DATA, VER_DATA, image))

        # Make predictions
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(os.path.join(APP_DATA, VER_DATA)))
    verified = verification > verification_threshold

    return results, verified

# OpenCV real time verification
cap = cv2.VideoCapture('/dev/video0')

# While loop for testing with OpenCV
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120+250,200:200+250, :]

    cv2.imshow('Verification', frame)
    # Verification trigger
    if cv2.waitKey(10) & 0xFF == ord('v'):
        # Save input image 
        cv2.imwrite(os.path.join(APP_DATA, INP_IMG, 'input_image.jpg'), frame)
        input_img = preprocess(os.path.join(APP_DATA, INP_IMG, 'input_image.jpg'))
        # Verification function
    
    if cv2.waitKey(10) & 0xFF == ord('v'):
        results, verified = verify(model,  0.5, 0.5)
        print(verified)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

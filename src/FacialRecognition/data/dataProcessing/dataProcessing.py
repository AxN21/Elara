# Import Standard dependecies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# import tenserflow dependecies - Functional API
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
import uuid


# Define constants
ANC_PATH = "/home/axn/Desktop/Python projects/Elara/src/FacialRecognition/data/anchor"
POS_PATH = "/home/axn/Desktop/Python projects/Elara/src/FacialRecognition/data/positive"
NEG_PATH = "/home/axn/Desktop/Python projects/Elara/src/FacialRecognition/data/negative"

# Improvements with data augmentation (still in testing)
'''def data_aug(img):
    data = []
    for i in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,3))
        # img = tf.image.stateless_random_crop(img, size=(20,20,3), seed=(1,2))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9,upper=1, seed=(np.random.randint(100),np.random.randint(100)))
            
        data.append(img)
    
    return data'''

'''for file_name in os.listdir(os.path.join(POS_PATH)):
    img_path = os.path.join(POS_PATH, file_name)
    img = cv2.imread(img_path)
    augmented_images = data_aug(img) 
    
    for image in augmented_images:
        cv2.imwrite(os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())'''

# PREPROCESSING - SCALE AND RESIZE DATA
anchor = tf.data.Dataset.list_files(ANC_PATH+'/*.jpg').take(1000)
positive = tf.data.Dataset.list_files(POS_PATH+'/*.jpg').take(1000)
negative = tf.data.Dataset.list_files(NEG_PATH+'/*.jpg').take(1000)

# Function for preproccesing images
def preprocess(file_path):

    # Read an image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100, 100))
    # Scale image to be between 0 and 1
    img = img / 255.0
    # Return image

    return img

# (anchor, positive) => 1,1,1,1,1
# (anchor, negative) => 0,0,0,0,0
# CREATE LABELLED DATASET
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

# BUILD TRAIN AND TEST PARTITION
def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)


# BUILD DATALOADER PIPELINE
data = data.map(preprocess_twin)
data = data.cache()

# Shuffling (consistent across anchor, positive, negative)
data = data.shuffle(buffer_size=10000)

# TRAINING PARTITION
train_data = data.take(round(len(data) * 0.7))
train_data = train_data.shuffle(buffer_size=1000)  # Shuffle after batching
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# TESTING PARTITION
test_data = data.skip(round(len(data) * 0.7))
test_data = test_data.take(round(len(data) * 0.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


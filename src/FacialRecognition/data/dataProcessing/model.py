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
from keras.metrics import Precision, Recall

# Import Training dataset
from dataProcessing import train_data

# Function to create the embedding layer
def make_embedding():
    inp = Input(shape=(100, 100, 3), name = 'input_image')

    # First Block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(pool_size=(2,2), padding='same')(c1)
    # Second Block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(pool_size=(2,2), padding='same')(c2)

    # Third Block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(pool_size=(2,2), padding='same')(c3)

    # Final Embedding Block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name= 'embedding')

embedding = make_embedding()

# Build Distance Layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


# Function to create the embedding layer
def make_siamese_model():

    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100, 100, 3))

    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # Classification Layer 
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


siamese_model = make_siamese_model()
siamese_model.summmary()

# defining the loss function
binary_cross_loss = tf.losses.BinaryCrossentropy()
# define adam optimizer
opt = tf.keras.optimizers.Adam(1e-4) # 0.0001
siamese_model.compile(optimizer=opt, loss=binary_cross_loss, metrics=['accuracy'])

# define checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt') # To reload from the checkpoint you can use model.load('path_to_checkpoint') This will load the pre trained wheights into the existing model.
checkpoint = tf.train.Checkpoint(opt = opt, siamese_model= siamese_model)


# Build and Train Step Function
@tf.function
def train_step(batch):

    # Record all of our operations
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative images
        X = batch[:2]
        # Get label
        y = batch[2]

        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)

    # Calculate gradients 
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    return loss

# defining training loop
def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS + 1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

        # Creating a metric object
        r = Recall()
        p = Precision()

        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())

        # Save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


EPOCHS = 50
train(train_data, EPOCHS)
siamese_model.save('siamese.h5')
test_input, test_val, y_true = train_data.as_numpy_iterator().next()
y_hat = siamese_model.predict([test_input, test_val])

ss= [1 if prediction > 0.7 else 0 for prediction in y_hat]
print(y_true)
num_samples = len(y_true)

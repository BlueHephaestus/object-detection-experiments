import h5py
import numpy as np
import random

import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2

def recursive_get_type(x):
    try:
        return recursive_get_type(x[0])
    except:
        return type(x)

def to_categorical(vec, width):
    """
    Arguments:
        vec: Vector of indices
        width: Width of our categorical matrix, should be the max value in our vector (but could also be larger if you want).

    Returns:
        A one hot / categorical matrix, so that each entry is the one-hot vector of the index, e.g.
            2 (with width = 4) -> [0, 0, 1, 0]
    """
    categorical_mat = np.zeros((len(vec), width))
    categorical_mat[np.arange(len(vec)), vec] = 1
    return categorical_mat

def get_new_balanced_batch(x, y, sample_n, positive_n):

    while True:

        batch_size = int(positive_n*2)
        """
        We don't want too large of a batch size
        """
        batch_size = min(batch_size, 200)
        x_batch = np.zeros((batch_size, 512, 512, 3))
        y_batch = np.zeros((batch_size, 2))

        """
        First get positive examples by picking batch_size/2 from our first positive_n samples.
        """
        positive_indices = random.sample(range(positive_n), batch_size/2)
        for i, positive_i in enumerate(positive_indices):
            x_batch[i] = x[positive_i]
            y_batch[i] = y[positive_i]

        """
        Then randomly choose batch_size/2 number of indices after our positive examples, 
            so we get an equal number of random samples from our negatives.
        We use positive_n so we don't accidentally choose positive samples where we need negative ones
        We use random.sample for this.
        """
        negative_indices = random.sample(range(positive_n, sample_n), batch_size/2)
        for i, negative_i in enumerate(negative_indices):
            x_batch[(batch_size/2)+i] = x[negative_i]
            y_batch[(batch_size/2)+i] = y[negative_i]

        yield x_batch, y_batch

def train_model(archive_dir, model_dir):
    """
    Arguments:
        archive_dir: An h5py archive directory with an x, x_shape, y, and y_shape dataset; where
            x: training data np array of shape (n, f) where n is the number of samples and f is the number of features.
            x_shape: shape of the x array
            y: training data np array of shape (n,) where n is the number of samples
            y_shape: shape of the y array
        model_dir: Filepath to store our model

    Returns:
        Saves a Keras model trained on this data to the model filepath, 
    """
    """
    Load our X and Y data,
        using a memmap for our x data because it may be too large to hold in RAM,
        and loading Y as normal since this is far less likely 
            also, using a memmap for Y when it is very unnecessary would likely impact performance significantly.

    I noticed that assigning the memmap all at once would often still be too large to hold in RAM,
        so we step through the archive and assign it in sections at a time.
    We step through and assign sections based on memmap_step.
    """
    with h5py.File(archive_dir, "r", chunks=True, compression="gzip") as hf:

        x = np.array(hf.get("x"))
        y = np.array(hf.get("y")).astype(int)

        """
        Assumes our x array has the positive_n samples first, unshuffled,
        and the remaining samples are the negative samples.
        """
        sample_n = len(x)
        positive_n = np.sum(y==1)

        y = to_categorical(y, 2)

        input_shape = [hf.get("x_shape")[1]]
        #input_shape = [512,512,3]

        print "Testing model 1"
        loss = "binary_crossentropy"
        optimizer = Adam()
        regularization_rate = 1e-4
        epochs = 10
        mini_batch_size=100

        model = Sequential()

        print input_shape
        #model.add(Dense(2048, input_shape=input_shape, activation="sigmoid", kernel_regularizer=l2(regularization_rate)))
        model.add(Dense(512, input_shape=input_shape, activation="sigmoid", kernel_regularizer=l2(regularization_rate)))
        model.add(Dense(128, activation="sigmoid", kernel_regularizer=l2(regularization_rate)))
        model.add(Dense(2, activation="softmax", kernel_regularizer=l2(regularization_rate)))
        model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

        #model.fit_generator(get_new_balanced_batch(x, y, sample_n, positive_n), steps_per_epoch=100, epochs=epochs)
        #model.fit_generator(get_new_balanced_batch(x, y, sample_n, positive_n), steps_per_epoch=100, epochs=epochs)
        model.fit(x, y, epochs=epochs, batch_size=mini_batch_size)

        predictions = model.predict(x)
        print np.sum(np.argmax(predictions, axis=1)==1)
        print np.sum(np.argmax(predictions, axis=1)==0)
        acc = np.sum(np.argmax(predictions, axis=1) == np.argmax(y, axis=1)) / float(sample_n)
        print "Raw Acc: ", acc

        """
        Save our model to the model filepath
        """
        print "Saving Model..."
        model.save("%s.h5" % (model_dir))

train_model("hog_samples.h5", "balanced_type1_detection_model_1")

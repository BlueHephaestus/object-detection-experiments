import cv2
import h5py
import numpy as np

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

def train_svm(archive_dir, svm_dir):
    """
    Arguments:
        archive_dir: An h5py archive directory with an x, x_shape, y, and y_shape dataset; where
            x: training data np array of shape (n, f) where n is the number of samples and f is the number of features.
            x_shape: shape of the x array
            y: training data np array of shape (n,) where n is the number of samples
            y_shape: shape of the y array

    Returns:
        Saves an OpenCV SVM trained on this data to the svm_dir, 
            using either the default parameters or ones set in the code of this function
    """
    """
    Load our X and Y data,
        using a memmap for our x data because it may be too large to hold in RAM,
        and loading Y as normal since this is far less likely 
            also, using a memmap for Y when it is very unnecessary would likely impact performance significantly.
    """
    with h5py.File(archive_dir, "r", chunks=True, compression="gzip") as hf:
        x_shape = tuple(hf.get("x_shape"))
        x = np.memmap("x.dat", dtype="float32", mode="w+", shape=x_shape)
        memmap_step = 1000
        hf_x = hf.get("x")
        for i in range(0, x_shape[0], memmap_step):
            x[i:i+memmap_step] = hf_x[i:i+memmap_step]
            print i
        #x[:] = hf.get("x")[:]
        #x = hf.get("x")

        y = np.array(hf.get("y")).astype(int)

        """
        Initialize a new opencv SVM
        """
        svm = cv2.ml.SVM_create()

        """
        Set parameters for our svm
        """
        #svm_params = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC)

        """
        Train SVM on our input data and output data (X and Y)
        """
        svm.train(x, cv2.ml.ROW_SAMPLE, y)
        #svm.train_auto(x, y, None, None, params=svm_params)

        """
        Then get our actual output vector via flattening of our svm's predictions on training data
        """
        actual_output = svm.predict(x)[1].flatten()

        """
        Use this to compute accuracy, via (the number of output predictions which match our desired predictions) / (total predictions)
        """
        acc = np.sum(actual_output == y) /float(len(y))
        print "SVM Training Accuracy: %.02f" % acc

        """
        Save our model to the svm_dir
        """
        svm.save(svm_dir)

train_svm("hog_samples.h5", "type1_detection_svm.xml")

import cv2

import numpy as np

import sklearn
from sklearn import datasets

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

iris = datasets.load_iris()
"""
Get only first 100, so that it's a binary classification problem
"""
X = iris.data
X = X.astype(np.float32)
Y = iris.target
#Y = Y[:, np.newaxis]
#Y = to_categorical(Y, np.max(Y)+1)

svm = cv2.ml.SVM_create()

X = X.astype(np.float32)
#Y = Y.astype(np.float64)
#svm_params = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC)
#svm.train(X, Y)
#svm.train_auto(X, Y, None, None, params=svm_params)
svm.train(X, cv2.ml.ROW_SAMPLE, Y, k_fold=5)
#print dir(svm)
#print svm.params()
A = svm.predict(X)[1]
print A
print A.shape
A = np.reshape(A, (-1)).astype(int)
acc = np.sum((A == Y).astype(int))/float(len(Y))
print acc
print X.shape
print A.shape
print svm.getSupportVectors().shape
#hog = cv2.HOGDescriptor()


#hog.setSVMDetector(cv2.

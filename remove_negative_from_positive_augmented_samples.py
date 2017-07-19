import h5py
import numpy as np
import cv2

def read_new(archive_dir):
    with h5py.File(archive_dir, "r", chunks=True, compression="gzip") as hf:
        """
        Load our X data the usual way,
            using a memmap for our x data because it may be too large to hold in RAM,
            and loading Y as normal since this is far less likely 
                -using a memmap for Y when it is very unnecessary would likely impact performance significantly.
        """
        x_shape = list(hf.get("x_shape"))
        x_shape[0] = x_shape[0]-27
        x_shape = tuple(x_shape)
        print x_shape
        x = np.memmap("x.dat", dtype="float32", mode="r+", shape=x_shape)
        memmap_step = 1000
        hf_x = hf.get("x")
        for i in range(27, x_shape[0]+27, memmap_step):
            x[i-27:i-27+memmap_step] = hf_x[i:i+memmap_step]
            print i
        y = np.ones((x_shape[0]))
    return x, y

def write_new(archive_dir, x, y):
    with h5py.File(archive_dir, "w", chunks=True, compression="gzip") as hf:
        hf.create_dataset("x", data=x)
        hf.create_dataset("x_shape", data=x.shape)
        hf.create_dataset("y", data=y)
        hf.create_dataset("y_shape", data=y.shape)

"""
Just gets rid of the negatives by only reading the positives, then writing them to replace the existing archive
"""
archive_dir="positive_augmented_samples.h5"
x,y = read_new(archive_dir)
write_new(archive_dir, x, y)


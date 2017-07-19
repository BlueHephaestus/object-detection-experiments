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
        x_shape = tuple(hf.get("x_shape"))
        print x_shape
        x = np.memmap("x.dat", dtype="float32", mode="r+", shape=x_shape)
        memmap_step = 1000
        hf_x = hf.get("x")
        for i in range(0, x_shape[0], memmap_step):
            x[i:i+memmap_step] = hf_x[i:i+memmap_step]
            print i
            break
        y = np.array(hf.get("y")).astype(int)
    return x, y

x,y = read_new("positive_augmented_samples.h5")
for i, sample in enumerate(x[:100]):
    sample = np.reshape(sample, (128, 128, 3))
    cv2.imwrite("tmp/%i.jpg"%i, sample)
    print i, y[i] == 0

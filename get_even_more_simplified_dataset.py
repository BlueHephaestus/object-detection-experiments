import h5py
import numpy as np

def read_new(archive_dir):
    with h5py.File(archive_dir, "r", chunks=True, compression="gzip") as hf:
        """
        Load our X data the usual way,
            using a memmap for our x data because it may be too large to hold in RAM,
            and loading Y as normal since this is far less likely 
                -using a memmap for Y when it is very unnecessary would likely impact performance significantly.
        """
        x_shape = tuple(hf.get("x_shape"))
        x = np.memmap("x.dat", dtype="uint8", mode="r+", shape=x_shape)
        memmap_step = 1000
        hf_x = hf.get("x")
        for i in range(0, x_shape[0], memmap_step):
            x[i:i+memmap_step] = hf_x[i:i+memmap_step]
            print i
        y = np.array(hf.get("y"))
    return x,y

def write_new(new_dir, x, y):
    with h5py.File(new_dir, "w", chunks=True, compression="gzip") as hf:
        hf.create_dataset("x", data=x)
        hf.create_dataset("x_shape", data=x.shape)
        hf.create_dataset("y", data=y)
        hf.create_dataset("y_shape", data=y.shape)

x,y = read_new("augmented_samples.h5")
new_n = 1956*2
new_x = np.memmap("new_x.dat", dtype="uint8", mode="w+", shape=(new_n, 512, 512, 3))
new_y = np.zeros((new_n,))

new_x[:1956] = x[:1956]
new_y[:1956] = 1.0
new_x[1956:] = x[1956:new_n]
new_y[1956:] = 0.0
        
write_new("samples.h5", new_x, new_y)



"""
Check to make sure we have our positives at the beginning
We don't
Manually put positives that aren't in the beginning in a new file
and don't add the last negatives either
"""

import numpy as np
import h5py

initial_archive_dir = "smaller_samples.h5"
new_archive_dir = "smaller_positive_samples.h5"

with h5py.File(initial_archive_dir, "r", chunks=True, compression="gzip") as hf:
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

"""
Make sure we have one negative example to make the transformation go much easier
"""
"""
Also there are very few positive examples so this shouldn't kill memory
"""
new_x = [x[-1]]
new_y = [0]

for sample_i in range(x_shape[0]):
    if y[sample_i] == 1:
        #sample is positive, concatenate
        new_x.append(x[sample_i])
        new_y.append(1)

new_x = np.array(new_x)
new_y = np.array(new_y)

with h5py.File(new_archive_dir, "w", chunks=True, compression="gzip") as hf:
    hf.create_dataset("x", data=new_x)
    hf.create_dataset("x_shape", data=new_x.shape)
    hf.create_dataset("y", data=new_y)
    hf.create_dataset("y_shape", data=new_y.shape)

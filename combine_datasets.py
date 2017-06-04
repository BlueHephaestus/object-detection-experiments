import numpy as np
import h5py

archive_dir_1 = "smaller_samples.h5"
archive_dir_2 = "smaller_positive_augmented_samples.h5"

combined_archive_dir = "smaller_augmented_samples.h5"

with h5py.File(archive_dir_1, "r", chunks=True, compression="gzip") as hf:
    x_shape_1 = tuple(hf.get("x_shape"))
with h5py.File(archive_dir_2, "r", chunks=True, compression="gzip") as hf:
    x_shape_2 = tuple(hf.get("x_shape"))

combined_n = x_shape_1[0] + x_shape_2[0]
combined_dims = [combined_n]
combined_dims.extend(x_shape_1[1:])
combined_dims = tuple(combined_dims)

combined_x = np.memmap("combined_x.dat", dtype="uint8", mode="w+", shape=combined_dims)
combined_y = np.zeros((combined_n,))
"""
Then we load our data now that we have a memmap ready
"""
with h5py.File(archive_dir_1, "r", chunks=True, compression="gzip") as hf:
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

combined_x[:x_shape[0]] = x
combined_y[:x_shape[0]] = y

"""
And do that again with the other one
"""
with h5py.File(archive_dir_2, "r", chunks=True, compression="gzip") as hf:
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

combined_x[-x_shape[0]:] = x
combined_y[-x_shape[0]:] = y


"""
Then write it
"""
with h5py.File(combined_archive_dir, "w", chunks=True, compression="gzip") as hf:
    hf.create_dataset("x", data=combined_x)
    hf.create_dataset("x_shape", data=combined_x.shape)
    hf.create_dataset("y", data=combined_y)
    hf.create_dataset("y_shape", data=combined_y.shape)

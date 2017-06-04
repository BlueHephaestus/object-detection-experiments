import h5py
import numpy as np

with h5py.File("samples.h5", "r", chunks=True, compression="gzip") as hf:
    y = np.array(hf.get("y"))
print np.sum(y==0)
print np.sum(y==1)

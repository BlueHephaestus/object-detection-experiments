import cv2
import numpy as np
import time

def recursive_get_type(x):
    try:
        return recursive_get_type(x[0])
    except:
        return type(x)

start_time = time.time()

win_size = (512, 512)
block_size = (16,16)
block_stride = (8,8)
cell_size = (8,8)
nbins = 9
deriv_aperture = 1
win_sigma = 4.
histogram_norm_type = 0
l2_hys_threshold = 2.0000000000000001e-01
gamma_correction = False
n_levels = 64
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins, deriv_aperture, win_sigma, histogram_norm_type, l2_hys_threshold, gamma_correction, n_levels)

img = np.floor(np.random.rand(512,512,3)*255).astype(np.uint8)
descriptors = hog.compute(img)

print descriptors
print descriptors.shape
print recursive_get_type(descriptors)
print time.time() - start_time

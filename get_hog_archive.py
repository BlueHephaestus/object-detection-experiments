import cv2
import numpy as np
import h5py

def get_hog_archive(archive_dir, hog_archive_dir, img_h=512, img_w=512, rgb=True):
    with h5py.File(archive_dir, "r", chunks=True, compression="gzip") as hf:
        """
        Load our X and Y data,
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
    Now feed all our samples through a HOG Descriptor, to get feature descriptors for each sample.
    """
    """
    We initialize our hog descriptor,
    """
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

    """
    print x[0].shape
    a = x[0]
    a = np.reshape(a, (img_h, img_w, -1))
    print a.shape
    """
    """
    We reshape our x to be an image and convert it to type np.uint8, 
        so that we can properly put it through our HOG / so it will be compatible with OpenCV's HOG,
    """
    if rgb:
        x = np.reshape(x, (-1, img_h, img_w, 3))
    else:
        x = np.reshape(x, (-1, img_h, img_w))

    """
    Then we feed a sample through to get what our output vector dimensions will be
    For some reason opencv returns this as an array of shape (n, 1) instead of just (n,),
        So we have to reshape this afterwards.
    """
    hog_output_shape = hog.compute(x[0]).shape

    """
    Using this, create a final array to store our x values in,
        (with the sample_n obtained from the first dimension of x)
        whereas y is same as the original
    """
    hog_x_dims = [x_shape[0]]
    hog_x_dims.extend(hog_output_shape)
    #hog_x = np.zeros(hog_x_dims)
    hog_x = np.memmap("hog_x.dat", dtype="float32", mode="w+", shape=tuple(hog_x_dims))

    """
    And finally loop through all our x samples and feed them to our HOG
    """
    for sample_i, sample in enumerate(x):
        hog_x[sample_i] = hog.compute(sample)

    """
    And reshape our hog_x after finished, since it will have an extraneous dimension
    """
    hog_x = np.reshape(hog_x, (hog_x_dims[0], hog_x_dims[1]))

    
    """
    We then create an h5py file in hog_archive_dir, and save our new samples to it
    """
    with h5py.File(hog_archive_dir, "w", chunks=True, compression="gzip") as hf:
        hf.create_dataset("x", data=hog_x)
        hf.create_dataset("x_shape", data=hog_x.shape)
        hf.create_dataset("y", data=y)
        hf.create_dataset("y_shape", data=y.shape)

get_hog_archive("phase2_smaller_augmented_samples.h5", "hog_samples.h5")

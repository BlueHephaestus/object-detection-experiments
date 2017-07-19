import sys
import numpy as np
import cv2, h5py

sys.path.append("../tuberculosis_project/lira_static")
import object_detection_handler
from object_detection_handler import *

a = ObjectDetector("type1_detection_model_1", "../tuberculosis_project/lira/lira2/saved_networks")

def filepath_gen(file_dir):
    for path in os.walk(file_dir):
        dir, b, fnames = path
        for fname in fnames:
            fpath = dir + os.sep + fname
            yield fpath

def add_detected_false_positives(f):
    img = cv2.imread(f)
    img = img.astype(np.uint8)
    img_detected_bounding_rects = a.generate_bounding_rects(img)
    """
    If we detected any, return the associated sample as a "hard" negative sample, 
        a negative which was predicted as a positive example. hard negative = false positive.

    We want to return a list of these results, so the list will be of size n x win_size x win_size x 3
    """
    results = []
    for (x1,y1,x2,y2) in img_detected_bounding_rects:
        results.append(np.reshape(cv2.resize(img[y1:y2,x1:x2], (512,512)), (-1, 3)))
    return np.array(results)




"""
We loop through each slide, get false positives on each one (if any),
    adding them to a new array.
Once done, we open our old array and add our new samples onto the end manually
This is both much simpler because we can create one separate new memmap 
    for the final combined result,
And we don't have to waste memory by holding the initial memmap there while creating
    our secondary one.
"""

false_positives = np.array([])
for i, fpath in enumerate(filepath_gen("hnm_phase_2_samples")):
    detected_false_positives = add_detected_false_positives(fpath)
    if len(false_positives) == 0:
        """
        If we don't have any (false positives is empty), just set our false positives to our new detected false positives
        """
        false_positives = detected_false_positives
    else:
        """
        Otherwise add our detected false positives to the end
        """
        if len(detected_false_positives) != 0:
            false_positives = np.concatenate((false_positives, detected_false_positives), axis=0)
    print detected_false_positives.shape, false_positives.shape

print false_positives.shape

"""
Now we have all of our false positives. 
We want to concatenate these and our original negatives such that false positives + original negatives = new negatives,
but since this is gonna be a big array, we need to create a memmap which matches the shape of our new negatives.
So we get the shape of our original negatives, and add the shape of our false positives
"""
initial_archive_dir = "augmented_samples.h5"
new_archive_dir = "phase2_augmented_samples.h5"

with h5py.File(initial_archive_dir, "r", chunks=True, compression="gzip") as hf:
    x_shape = hf.get("x_shape")
    new_negatives_n = false_positives.shape[0] + x_shape[0]
    new_negative_dims = [new_negatives_n]
    new_negative_dims.extend(false_positives.shape[1:])
    new_negative_dims = tuple(new_negative_dims)

new_x = np.memmap("new_x.dat", dtype="uint8", mode="w+", shape=new_negative_dims)
new_y = np.zeros((new_negatives_n,))

with h5py.File(initial_archive_dir, "r", chunks=True, compression="gzip") as hf:
    """
    Load our X data the usual way,
        using a memmap for our x data because it may be too large to hold in RAM,
        and loading Y as normal since this is far less likely 
            -using a memmap for Y when it is very unnecessary would likely impact performance significantly.
    """
    x_shape = tuple(hf.get("x_shape"))
    x = np.memmap("x.dat", dtype="uint8", mode="r+", shape=(x_shape[0], 262144, 3))
    memmap_step = 1000
    hf_x = hf.get("x")
    for i in range(0, x_shape[0], memmap_step):
        x[i:i+memmap_step] = np.reshape(hf_x[i:i+memmap_step], (-1, 262144, 3))
        print i
    y = np.array(hf.get("y"))

"""
Ok so now we have our old x, old y, a new x memmap ready, our new negative samples, and we know those new negatives have y = 0 as their y values.
We manually insert our samples in their correct places.
"""
new_x[:x_shape[0]] = x
new_x[-len(false_positives):] = false_positives
new_y[:x_shape[0]] = y
new_y[-len(false_positives):] = 0

with h5py.File(new_archive_dir, "w", chunks=True, compression="gzip") as hf:
    hf.create_dataset("x", data=new_x)
    hf.create_dataset("x_shape", data=new_x.shape)
    hf.create_dataset("y", data=new_y)
    hf.create_dataset("y_shape", data=new_y.shape)

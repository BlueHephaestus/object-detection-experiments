import os
import cv2 
import numpy as np

result_dir = "checks"
for path in os.walk("../tuberculosis_project/lira/lira1/data/full_test_slides_set"):
    dir, b, fnames = path
    for i, fname in enumerate(fnames):
        fpath = dir + os.sep + fname
        img = cv2.imread(fpath)
        fname = fname.split(".")[0]
        img = cv2.resize(img, (0, 0), fx=0.05, fy=0.05)
        print i, fname, img.shape
        cv2.imwrite("%s%s%s.jpg" % (result_dir, os.sep, fname),img)


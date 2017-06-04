import os
import cv2 
import numpy as np

shapes = []
for path in os.walk("./raw_data"):
    dir, b, fnames = path
    for i, fname in enumerate(fnames):
        fpath = dir + os.sep + fname
        img = cv2.imread(fpath)
        img = cv2.resize(img, (512, 512))
        cv2.imwrite("./data/positives/%s.png" % str(i), img)
        print img.shape

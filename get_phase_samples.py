import os
import shutil

import numpy as np

def filename_gen(file_dir):
    for path in os.walk(file_dir):
        dir, b, fnames = path
        for fname in fnames:
            yield fname

positives = []
with open("tmp.txt") as f:
    for line in f:
        line = line.strip()
        positives.append("%s.svs" % line)
#print positives

negatives = []
img_dir = "../tuberculosis_project/lira/lira1/data/full_test_slides_set"
for slide in filename_gen(img_dir):
    slide = slide.strip()
    if slide not in positives:
        """
        Slide is a negative
        """
        negatives.append(slide)
"""
then we divide into n sized sublists from our negatives list
"""
n = 40
negatives = np.array(negatives)
negatives = np.split(negatives, [n+i*n for i in range(len(negatives)/n)])
for phase_i, phase_negatives in enumerate(negatives):
    for negative in phase_negatives:
        print phase_i, negative
        negative_fpath = "%s%s%s" % (img_dir, os.sep, negative)
        shutil.copy(negative_fpath, "hnm_phase_%i_samples%s%s" % (phase_i+1, os.sep, negative))



"""now we need to get all the ones which are not positives, and move 40 to one 40 to another"""
"""we loop through the full test slide set and compare the fnames"""
"""We copy all negatives from src to hnm_phase_x_samples"""

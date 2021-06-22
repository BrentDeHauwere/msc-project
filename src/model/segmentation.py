import cv2
import os
import glob
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

INPUT_SEGMENTATION_DIR = 'data/processed/segmentation/'

LTHIGH_SEGMENT_COLOUR = [0, 127, 255]


# iterate over each sequence (represented by a directory)
for seq_id in [x for x in os.listdir(INPUT_SEGMENTATION_DIR) if os.path.isdir(INPUT_SEGMENTATION_DIR + x)]:

    # retrieve segmentation filesnames and sort
    filenames = [os.path.basename(x) for x in glob.glob(
        f'{INPUT_SEGMENTATION_DIR}{seq_id}/*.npy')]
    filenames.sort()

    # iterate over segmentation files (one file per frame)
    for file in tqdm(filenames, desc=seq_id):

        # load segmentation file
        segm_arr = np.load(f'{INPUT_SEGMENTATION_DIR}{seq_id}/{file}')

        # exported with opencv (BGR), converted to RGB for pillow (RGB)
        segm_arr = segm_arr[:, :, [2, 1, 0]]

        Image.fromarray(segm_arr).show()

        x = 0

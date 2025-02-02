# %%
import math
import os
import glob
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

INPUT_SEQUENCES_DIR = 'data/processed/sequences/'
INPUT_POSE_ESTIMATION_DIR = 'data/processed/pose_estimation/'
OUTPUT_DIR = 'data/plots/'

N_POINTS = 25
POINT_LTHIGH = 12
POINT_LKNEE = 13
POINT_LANKLE = 14
POINT_NECK = 1
POINT_MIDHIP = 8


def calc_cos_body_part(seq_id, frame_id, keypoint1, keypoint2):

    with open(f'{INPUT_POSE_ESTIMATION_DIR}{seq_id}/{frame_id}_keypoints.json') as json_file:
        data = json.load(json_file)

    assert len(data['people']) == 1

    # for example: x_lknee
    x1 = data['people'][0]['pose_keypoints_2d'][keypoint1*3]
    # for example: y_lknee
    y1 = data['people'][0]['pose_keypoints_2d'][keypoint1*3 + 1]
    # for example: c_lknee
    c1 = data['people'][0]['pose_keypoints_2d'][keypoint1*3 + 2]
    # print(x1, y1, c1)

    # for example: x_lankle
    x2 = data['people'][0]['pose_keypoints_2d'][keypoint2*3]
    # for example: y_lankle
    y2 = data['people'][0]['pose_keypoints_2d'][keypoint2*3 + 1]
    # for example: c_lankle
    c2 = data['people'][0]['pose_keypoints_2d'][keypoint2*3 + 2]
    # print(x2, y2, c2)

    # angle in rad (- because x-axis is inverted)
    m = - math.atan2(y1-y2, x1-x2)

    # angle in degrees
    theta = math.degrees(m)

    # cosinus of angle (between thigh and x-axis)
    cos_theta = math.cos(m)

    return theta


def manual_slicing(frames):
    if seq_id == '009a017s04L':
        frames = frames[16:]
    elif seq_id == 'fyc-00_3':
        frames = frames[6:]
    elif seq_id == 'syj-00_3':
        frames = frames[2:]

    return frames


# generate plot for each body part
for bp_name, p1, p2 in [['left thigh', POINT_LTHIGH, POINT_LKNEE],
                        ['left leg', POINT_LKNEE, POINT_LANKLE],
                        ['torso', POINT_NECK, POINT_MIDHIP]]:

    # retrieve all the gait sequence ids
    for seq_id in [x for x in os.listdir(INPUT_SEQUENCES_DIR) if os.path.isdir(INPUT_POSE_ESTIMATION_DIR + x)]:

        # retrieve framenames and sort
        frames = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(
            f'{INPUT_SEQUENCES_DIR}{seq_id}/*.png')]
        frames.sort()

        # MANUAL SLICING OF SEQUENCE TO SYNCHRONISE GAIT CYCLE
        frames = manual_slicing(frames)

        # retrieve keypoints and calculate theta of thigh
        th = [calc_cos_body_part(seq_id, f, p1, p2) for f in frames]
        th = gaussian_filter1d(th, sigma=1.6, truncate=2)

        # plot cos(th) per frame
        sns.scatterplot(x=range(len(th)),
                        y=th,
                        label='subject ' + seq_id)

    # plt.title(f'Inclination of {bp_name.title()}')
    plt.xlabel('x')
    plt.ylabel('θ')
    plt.legend(loc='upper right')
    plt.savefig(
        OUTPUT_DIR + f'inclination_of_{bp_name.replace(" ", "_")}')
    plt.show()
    plt.clf()
